"""
Fase 0 — Contrato del estudio.

Valida `contract.yaml` (completitud + coherencia con el tipo de problema), congela el snapshot
de datos y materializa la separación desarrollo / holdout y los folds de CV.

Uso:
    python scripts/contract.py init     --run runs/<p> [--type forecasting|classification|regression]
    python scripts/contract.py check    --run runs/<p>     # solo revisa; no escribe nada
    python scripts/contract.py validate --run runs/<p> [--new-version]

`validate` produce:
    snapshot.json            hash, esquema, filas y procedencia declarada del snapshot
    dev.parquet              datos de desarrollo (+ columnas `_t` periodo y `_entidad` si aplican)
    _sealed/holdout.parquet  holdout; SOLO lo abre holdout.py
    holdout_manifest.json    ids, hashes y política de separación (sin valores del target)
    splits.json              folds materializados
y aprueba la fase `contrato` en state.json.

Reglas:
  * POR_DEFINIR = decisión de negocio sin tomar => falla (no se rellena en silencio).
  * forecasting => tiempo, horizonte >= 1, esquema temporal, holdout temporal_final.
  * generalización a periodos futuros => esquema temporal; a entidades nuevas => grupos.
  * Nunca kfold/estratificado aleatorio si hay columna de tiempo (salvo entidades nuevas con grupos).
  * Toda columna usable debe declarar disponibilidad; `desconocida` exige excepción documentada.
  * Si el holdout ya se evaluó, cualquier re-validación exige `--new-version` y el holdout
    queda marcado como ya observado.

Límite: el holdout queda aislado operativamente (los scripts ordinarios no lo cargan), pero no
es inaccesible: quien tenga el snapshot original o el directorio _sealed/ puede leerlo.
"""
from __future__ import annotations

import argparse
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from common import (HarnessError, TABLE_EXT, ahora, atomic_write_json, cli, env_info, find_table, load_state,
                    log_event, read_json, read_table, rel_table, sha256_file, sha256_json, state_tx, write_table)
from metrics import METRICAS, compatibles

TEMPLATES = Path(__file__).resolve().parent / "templates"
PD = "POR_DEFINIR"
TIPOS = ("forecasting", "classification", "regression")
DISPONIBILIDAD = ("conocida_anticipada", "al_predecir", "con_retraso", "posterior", "desconocida")
GENERALIZACION = ("periodos_futuros", "entidades_nuevas", "entidades_conocidas")
ESQUEMAS = ("temporal", "grupos", "estratificado", "kfold")
HOLDOUTS = ("temporal_final", "grupos", "aleatorio_estratificado", "aleatorio")
COLS_HARNESS = ("_t", "_entidad")


# --------------------------------------------------------------------------- carga

def cargar_contrato(run) -> dict:
    p = Path(run) / "contract.yaml"
    if not p.exists():
        raise HarnessError(f"no existe {p}.", f"Corre `python scripts/contract.py init --run {run}` y complétalo.")
    try:
        c = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as e:
        raise HarnessError(f"contract.yaml no es YAML válido: {e}", "Corrige la sintaxis del archivo.")
    return c


def cargar_dev(run) -> pd.DataFrame:
    p = find_table(Path(run) / "dev")
    if p is None:
        raise HarnessError("no existen los datos de desarrollo materializados.",
                           f"Corre `python scripts/contract.py validate --run {run}`.")
    df = read_table(p)
    return df


def _g(c: dict, ruta: str, default=None):
    cur = c
    for k in ruta.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


# --------------------------------------------------------------------------- disponibilidad de features

def columnas_estructurales(c: dict) -> set[str]:
    d = c.get("datos", {}) or {}
    return {x for x in (d.get("id_fila"), d.get("target"), d.get("tiempo")) if x}


def estado_feature(c: dict, f: str) -> tuple[str, str]:
    """Clasifica una columna como feature: ('ok' | 'excluida' | 'violacion' | 'excepcion' | 'sin_declarar', motivo)."""
    excl = set(_g(c, "datos.columnas_excluidas", []) or [])
    if f in excl:
        return "excluida", "en datos.columnas_excluidas"
    info = (_g(c, "features", {}) or {}).get(f)
    if info is None:
        return "sin_declarar", "no tiene disponibilidad declarada en `features`"
    disp = info.get("disponibilidad") if isinstance(info, dict) else info
    h = int(_g(c, "prediccion.horizonte", 0) or 0)
    if disp == "posterior":
        return "violacion", "declarada `posterior`: se conoce después del momento de predicción"
    if disp == "con_retraso":
        k = info.get("retraso") if isinstance(info, dict) else None
        if k is None:
            return "sin_declarar", "`con_retraso` sin `retraso: k`"
        if int(k) < h:
            return "violacion", f"retraso {k} < horizonte {h}: la ventana cruza el momento de predicción"
        return "ok", f"con retraso {k}"
    if disp == "desconocida":
        if f in (_g(c, "excepciones_disponibilidad", {}) or {}):
            return "excepcion", "disponibilidad desconocida con excepción documentada"
        return "sin_declarar", "disponibilidad `desconocida` sin excepción documentada"
    lin = info.get("linaje") if isinstance(info, dict) else None
    if lin and lin.get("fuente") == _g(c, "datos.target") and int(lin.get("desfase", 0) or 0) < max(h, 1):
        return "violacion", (f"su linaje dice que se deriva del target con desfase {lin.get('desfase', 0)} "
                             f"(< {max(h, 1)}): no existiría al predecir")
    if disp in ("conocida_anticipada", "al_predecir"):
        return "ok", disp
    return "sin_declarar", f"disponibilidad inválida: {disp!r}"


def features_utilizables(c: dict, columnas) -> list[str]:
    estr = columnas_estructurales(c)
    return [f for f in columnas if f not in estr and f not in COLS_HARNESS and estado_feature(c, f)[0] in ("ok", "excepcion")]


# --------------------------------------------------------------------------- revisión del contrato

def revisar(c: dict, columnas: list[str] | None = None, clases: int | None = None) -> list[tuple[str, str]]:
    """Devuelve problemas (qué, cómo). Vacío = contrato completo y coherente."""
    P: list[tuple[str, str]] = []

    def falta(ruta, como=None):
        v = _g(c, ruta)
        if v is None or v == PD or v == "":
            P.append((f"`{ruta}` está sin definir.", como or f"Define `{ruta}` en contract.yaml (si es decisión de negocio, pregúntala)."))
            return True
        return False

    for r in ("estudio", "problem_type", "datos.ruta", "datos.procedencia", "datos.id_fila", "datos.target",
              "datos.unidad_observacion", "prediccion.momento", "prediccion.generalizacion",
              "validacion.esquema", "validacion.justificacion", "holdout.estrategia", "metrica.principal",
              "comparacion.margen_practico"):
        falta(r)
    if P:
        return P  # sin lo básico no tiene sentido revisar coherencia

    pt = c["problem_type"]
    if pt not in TIPOS:
        return [(f"problem_type inválido: {pt!r}.", f"Usa uno de {TIPOS}.")]
    d, pr, v, ho, m = c["datos"], c["prediccion"], c["validacion"], c["holdout"], c["metrica"]
    tiempo, entidad = d.get("tiempo"), d.get("entidad")
    gen, esq, est = pr["generalizacion"], v["esquema"], ho["estrategia"]
    h = int(pr.get("horizonte", 0) or 0)

    if gen not in GENERALIZACION:
        P.append((f"prediccion.generalizacion inválida: {gen!r}.", f"Usa uno de {GENERALIZACION}."))
    if esq not in ESQUEMAS:
        P.append((f"validacion.esquema inválido: {esq!r}.", f"Usa uno de {ESQUEMAS}."))
    if est not in HOLDOUTS:
        P.append((f"holdout.estrategia inválida: {est!r}.", f"Usa uno de {HOLDOUTS}."))
    if d.get("duplicados") not in ("accidentales", "legitimos"):
        P.append(("datos.duplicados debe ser `accidentales` o `legitimos`.",
                  "Decide si las filas idénticas son errores de captura (accidentales) u observaciones reales (legitimos)."))

    # métricas
    for nombre in [m["principal"], *(m.get("secundarias") or [])]:
        if nombre not in METRICAS:
            P.append((f"métrica desconocida: {nombre!r}.", f"Usa una de {sorted(METRICAS)}."))
        elif nombre not in compatibles(pt, clases):
            P.append((f"la métrica {nombre!r} no es compatible con {pt}" + (f" de {clases} clases" if clases else "") + ".",
                      f"Compatibles: {compatibles(pt, clases)}."))
    if m.get("agregacion", "media_folds") not in ("media_folds", "pooled"):
        P.append(("metrica.agregacion debe ser media_folds o pooled.", "Corrígelo en contract.yaml."))
    try:
        if float(c["comparacion"]["margen_practico"]) < 0:
            P.append(("comparacion.margen_practico no puede ser negativo.", "Usa 0 o un valor positivo en unidades de la métrica."))
    except (TypeError, ValueError):
        P.append(("comparacion.margen_practico debe ser numérico.", "Exprésalo en unidades de la métrica principal."))

    # coherencia del esquema de validación
    if pt == "forecasting":
        if not tiempo:
            P.append(("forecasting sin columna de tiempo.", "Define datos.tiempo."))
        if h < 1:
            P.append(("forecasting con horizonte < 1.", "Define prediccion.horizonte en periodos (>= 1)."))
        if gen != "periodos_futuros":
            P.append(("forecasting debe generalizar a periodos_futuros.", "Ajusta prediccion.generalizacion."))
    if gen == "periodos_futuros":
        if esq != "temporal":
            P.append((f"generalización a periodos futuros con esquema {esq!r}: el CV aleatorio mezcla futuro y pasado.",
                      "Usa validacion.esquema: temporal (con gap >= retraso_target)."))
        if est != "temporal_final":
            P.append(("generalización a periodos futuros con holdout no temporal.", "Usa holdout.estrategia: temporal_final."))
        if not tiempo:
            P.append(("generalización a periodos futuros sin columna de tiempo.", "Define datos.tiempo."))
    if gen == "entidades_nuevas":
        if not entidad:
            P.append(("generalización a entidades nuevas sin columna de entidad.", "Define datos.entidad."))
        if esq != "grupos":
            P.append((f"generalización a entidades nuevas con esquema {esq!r}: la misma entidad quedaría en train y validación.",
                      "Usa validacion.esquema: grupos."))
        if est != "grupos":
            P.append(("generalización a entidades nuevas con holdout no agrupado.", "Usa holdout.estrategia: grupos."))
    if tiempo and esq in ("estratificado", "kfold"):
        P.append((f"hay columna de tiempo ({tiempo}) y el esquema es {esq} aleatorio: hay dependencia temporal.",
                  "Usa esquema temporal (o grupos si generalizas a entidades nuevas). Si el tiempo no es relevante, quita datos.tiempo y justifícalo en decisions.md."))
    if entidad and esq in ("estratificado", "kfold") and gen != "entidades_conocidas":
        P.append(("hay entidad y CV aleatorio sin declarar generalización a entidades conocidas.",
                  "Usa esquema grupos, o declara generalizacion: entidades_conocidas si esa es la pregunta real."))
    if esq == "estratificado" and pt != "classification":
        P.append(("esquema estratificado solo aplica a clasificación.", "Usa kfold para regresión."))
    if esq == "temporal":
        if not v.get("tamano_validacion"):
            P.append(("esquema temporal sin validacion.tamano_validacion.", "Define cuántos periodos tiene cada ventana de validación."))
        elif pt == "forecasting" and int(v["tamano_validacion"]) > h:
            P.append((f"tamano_validacion ({v['tamano_validacion']}) > horizonte ({h}).",
                      "Cada ventana de validación debe caber en el horizonte declarado; usa tamano_validacion <= horizonte."))
        if int(v.get("gap", 0) or 0) < int(pr.get("retraso_target", 0) or 0):
            P.append(("validacion.gap < prediccion.retraso_target: el entrenamiento usaría targets aún no conocidos.",
                      "Usa gap >= retraso_target."))
        if int(v.get("repeticiones", 1) or 1) != 1:
            P.append(("el esquema temporal no admite repeticiones.", "Usa repeticiones: 1."))
        if est == "temporal_final" and not ho.get("periodos"):
            P.append(("holdout temporal sin holdout.periodos.", "Define cuántos periodos finales se sellan."))
    if est in ("grupos", "aleatorio_estratificado", "aleatorio"):
        fr = ho.get("fraccion")
        if fr is None or not (0 < float(fr) < 1):
            P.append(("holdout.fraccion debe estar entre 0 y 1.", "Usa p. ej. 0.2."))
    if est == "aleatorio_estratificado" and pt != "classification":
        P.append(("holdout estratificado solo aplica a clasificación.", "Usa aleatorio."))
    if m.get("umbral", {}).get("politica", "fija") not in ("fija", "ajustada_interna"):
        P.append(("metrica.umbral.politica debe ser fija o ajustada_interna.", "Corrígelo en contract.yaml."))

    # disponibilidad de columnas
    if columnas is not None:
        for col in (d.get("id_fila"), d.get("target"), tiempo, entidad):
            if col and col not in columnas:
                P.append((f"la columna {col!r} declarada en el contrato no existe en los datos.", "Corrige el nombre en contract.yaml."))
        for col in [*(d.get("columnas_excluidas") or []), *(c.get("features") or {})]:
            if col not in columnas:
                P.append((f"la columna {col!r} del contrato no existe en los datos.", "Corrige el nombre o quítala del contrato."))
        estr = columnas_estructurales(c)
        for col in columnas:
            if col in estr or col in COLS_HARNESS:
                continue
            e, motivo = estado_feature(c, col)
            if e == "sin_declarar":
                P.append((f"columna {col!r}: {motivo}.",
                          f"Declara `features.{col}.disponibilidad` (conocida_anticipada | al_predecir | con_retraso+retraso | posterior), "
                          f"o exclúyela, o documenta la excepción en `excepciones_disponibilidad.{col}`."))
        for col, just in (c.get("excepciones_disponibilidad") or {}).items():
            if not isinstance(just, str) or len(just.strip()) < 20:
                P.append((f"excepción de disponibilidad de {col!r} sin justificación suficiente.",
                          "Escribe por qué no se puede resolver y qué riesgo se acepta (>= 20 caracteres)."))
    return P


def eval_hash(c: dict, hash_datos: str) -> str:
    """Hash de lo que define la evaluación: si cambia, los resultados previos dejan de ser comparables."""
    claves = {
        "problem_type": c.get("problem_type"), "datos": {k: _g(c, f"datos.{k}") for k in ("id_fila", "target", "entidad", "tiempo", "duplicados")},
        "prediccion": {k: _g(c, f"prediccion.{k}") for k in ("horizonte", "retraso_target", "generalizacion")},
        "validacion": c.get("validacion"), "holdout": c.get("holdout"),
        "metrica": c.get("metrica"), "datos_hash": hash_datos,
    }
    return sha256_json(claves)


# --------------------------------------------------------------------------- materialización

def _leer_snapshot(run, c) -> tuple[pd.DataFrame, Path]:
    p = Path(run) / c["datos"]["ruta"]
    if not p.exists():
        raise HarnessError(f"no existe el snapshot de datos {p}.",
                           "Exporta los datos (CSV o parquet) a esa ruta. Si vienen de SQL, expórtalos primero y documenta la query en datos.procedencia.")
    return read_table(p), p


def _periodos(serie: pd.Series) -> pd.Series:
    s = serie
    if s.dtype == object:
        try:
            s = pd.to_datetime(s)
        except (ValueError, TypeError):
            pass
    u = np.sort(s.unique())
    return s.map({v: i for i, v in enumerate(u)}).astype(int)


def preparar(df: pd.DataFrame, c: dict) -> tuple[pd.DataFrame, dict]:
    """Agrega columnas del harness y aplica la política de duplicados. Devuelve (df, info)."""
    d = c["datos"]
    info = {}
    idc = d["id_fila"]
    if df[idc].duplicated().any():
        n = int(df[idc].duplicated().sum())
        raise HarnessError(f"la columna id_fila {idc!r} tiene {n} valores repetidos.",
                           "id_fila debe identificar cada fila de forma única; corrige el export o define otra llave.")
    if df[d["target"]].isna().any():
        n = int(df[d["target"]].isna().sum())
        raise HarnessError(f"el target tiene {n} valores faltantes.",
                           "Decide (y registra en decisions.md) si esas filas se excluyen del snapshot o se imputan antes de exportar.")
    df = df.copy()
    contenido = [col for col in df.columns if col != idc]
    dup = df.duplicated(subset=contenido, keep="first")
    info["duplicados_exactos"] = int(dup.sum())
    info["ids_duplicados_eliminados"] = []
    if dup.any() and d["duplicados"] == "accidentales":
        info["ids_duplicados_eliminados"] = df.loc[dup, idc].astype(str).tolist()
        df = df.loc[~dup].copy()
    if d.get("tiempo"):
        df["_t"] = _periodos(df[d["tiempo"]])
    if d.get("entidad"):
        df["_entidad"] = df[d["entidad"]].astype(str)
    df[idc] = df[idc].astype(str)
    return df.reset_index(drop=True), info


def separar_holdout(df: pd.DataFrame, c: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    ho, d = c["holdout"], c["datos"]
    est = ho["estrategia"]
    seed = int(_g(c, "validacion.semilla_split", 42))
    info = {"estrategia": est}
    if est == "temporal_final":
        n = int(ho["periodos"])
        tmax = int(df["_t"].max())
        h_ini = tmax - n + 1
        gap = int(_g(c, "validacion.gap", 0) or 0)
        es_h = df["_t"] >= h_ini
        es_dev = df["_t"] <= h_ini - gap - 1
        info.update({"periodo_inicio": h_ini, "gap": gap, "filas_descartadas_gap": int((~es_h & ~es_dev).sum())})
        return df.loc[es_dev], df.loc[es_h], info
    rng = np.random.RandomState(seed)
    fr = float(ho["fraccion"])
    if est == "grupos":
        ents = np.array(sorted(df["_entidad"].unique()))
        k = max(1, int(round(len(ents) * fr)))
        sel = set(rng.choice(ents, size=k, replace=False))
        es_h = df["_entidad"].isin(sel)
        info["entidades_holdout"] = sorted(sel)
        return df.loc[~es_h], df.loc[es_h], info
    from sklearn.model_selection import train_test_split
    strat = df[d["target"]] if est == "aleatorio_estratificado" else None
    dv, hd = train_test_split(df, test_size=fr, random_state=seed, stratify=strat)
    return dv.sort_index(), hd.sort_index(), info


def _archivar_version(run: Path, st: dict) -> None:
    v = st.get("version_estudio", 1)
    dest = run / "versiones" / f"v{v}"
    dest.mkdir(parents=True, exist_ok=True)
    for nombre in ("holdout_result.json", "holdout_freeze.json", "final_report.md", "comparison.json", "review.json"):
        p = run / nombre
        if p.exists():
            shutil.copy2(p, dest / nombre)
            if nombre in ("holdout_result.json", "holdout_freeze.json", "final_report.md"):
                p.unlink()
    ho = st["holdout"]
    ho.setdefault("observaciones_previas", []).append({
        "version": v, "estado": ho.get("estado"), "candidato": (ho.get("candidato") or {}).get("run_id"),
        "congelado_sha": ho.get("congelado"),
        "fecha": ahora()})
    for f in ("holdout", "reporte"):   # sus artefactos quedaron archivados en versiones/v<n>/
        st["fases"].pop(f, None)
    st["version_estudio"] = v + 1
    st["holdout"] = {"estado": "sellado", "observaciones_previas": ho["observaciones_previas"],
                     "eventos": ho.get("eventos", [])}
    log_event(st, "nueva_version", version=v + 1, motivo="re-validación tras observar el holdout")


def validar(run, nueva_version: bool = False) -> dict:
    run = Path(run)
    c = cargar_contrato(run)
    probs = revisar(c)
    if probs:
        raise HarnessError("contrato incompleto o incoherente:\n" + "\n".join(f"  - {q}" for q, _ in probs),
                           "\n" + "\n".join(f"  - {w}" for _, w in probs))
    raw, snap = _leer_snapshot(run, c)
    clases = raw[c["datos"]["target"]].nunique() if c["problem_type"] == "classification" else None
    probs = revisar(c, list(raw.columns), clases)
    if probs:
        raise HarnessError("contrato incoherente con los datos:\n" + "\n".join(f"  - {q}" for q, _ in probs),
                           "\n" + "\n".join(f"  - {w}" for _, w in probs))

    st = load_state(run)
    if st["holdout"]["estado"] in ("en_curso", "evaluado") and not nueva_version:
        raise HarnessError(
            f"el holdout ya fue {'evaluado' if st['holdout']['estado'] == 'evaluado' else 'abierto'} en la versión {st['version_estudio']}.",
            "Cualquier cambio exige una nueva versión: `contract.py validate --new-version`. El holdout quedará marcado como "
            "ya observado y su resultado no será una estimación limpia; lo ideal es conseguir un holdout nuevo.")

    df, info_dup = preparar(raw, c)
    dev, hold, info_ho = separar_holdout(df, c)
    if len(hold) == 0 or len(dev) == 0:
        raise HarnessError("la separación desarrollo/holdout dejó un conjunto vacío.", "Ajusta holdout.periodos o holdout.fraccion.")
    if c["problem_type"] == "classification":
        faltan = set(df[c["datos"]["target"]].unique()) - set(dev[c["datos"]["target"]].unique())
        if faltan:
            raise HarnessError(f"las clases {faltan} no aparecen en desarrollo.", "Cambia la semilla o la estrategia del holdout.")
    dev = dev.reset_index(drop=True)
    from cv import make_splits, validar_splits
    splits = make_splits(dev, c)
    viol = validar_splits(dev, splits, c)
    if viol:
        raise HarnessError("los splits generados violan el contrato:\n" + "\n".join(viol), "Revisa validacion.* en contract.yaml.")

    with state_tx(run) as st:
        if st["holdout"]["estado"] in ("en_curso", "evaluado"):
            _archivar_version(run, st)
    dev_p = write_table(dev, run / "dev")
    hold_p = write_table(hold.reset_index(drop=True), run / "_sealed" / "holdout")
    atomic_write_json(run / "splits.json", splits)
    h_snap = sha256_file(snap)
    snapshot = {"ruta": c["datos"]["ruta"], "sha256": h_snap, "filas": int(len(raw)), "columnas": int(raw.shape[1]),
                "esquema": {k: str(v) for k, v in raw.dtypes.items()}, "procedencia": c["datos"]["procedencia"],
                "fecha_registro": ahora(), "entorno": env_info()}
    atomic_write_json(run / "snapshot.json", snapshot)
    idc = c["datos"]["id_fila"]
    manifest = {"estrategia": info_ho, "filas_holdout": int(len(hold)), "filas_dev": int(len(dev)),
                "ids_holdout": hold[idc].astype(str).tolist(), "sha256_holdout": sha256_file(hold_p),
                "duplicados": info_dup, "fecha": ahora(),
                "periodos_holdout": ([int(hold["_t"].min()), int(hold["_t"].max())] if "_t" in hold else None),
                "nota": "Aislamiento operativo: solo holdout.py abre _sealed/. No es una barrera de seguridad."}
    atomic_write_json(run / "holdout_manifest.json", manifest)
    eh = eval_hash(c, h_snap)
    rels = ["contract.yaml", c["datos"]["ruta"], "snapshot.json", dev_p.relative_to(run).as_posix(), "splits.json",
            "holdout_manifest.json", hold_p.relative_to(run).as_posix()]
    from gate import registrar
    with state_tx(run) as st:
        anterior = st["fases"].get("contrato", {}).get("eval_hash")
        if st["holdout"]["estado"] == "sin_sellar":
            st["holdout"]["estado"] = "sellado"
        log_event(st, "contrato_validado", eval_hash=eh, cambio_evaluacion=(anterior is not None and anterior != eh))
    registrar(run, "contrato", rels, eval_hash=eh, splits_hash=splits["hash"])
    return {"dev": len(dev), "holdout": len(hold), "folds": len(splits["folds"]), "eval_hash": eh,
            "duplicados_eliminados": len(info_dup["ids_duplicados_eliminados"]), "info_holdout": info_ho}


def contexto_eval(run) -> dict:
    """eval_hash y splits_hash vigentes (de la fase contrato aprobada)."""
    st = load_state(run)
    f = st["fases"].get("contrato")
    if not f:
        raise HarnessError("el contrato no está validado.", f"Corre `python scripts/contract.py validate --run {run}`.")
    return {"eval_hash": f["eval_hash"], "splits_hash": f["splits_hash"], "version": st.get("version_estudio", 1)}


def criterios(run):
    """Criterios de la fase contrato (los usa gate.py; la fase la aprueba `validate`)."""
    st = load_state(run)
    rec = st["fases"].get("contrato")
    if not rec:
        return [("el contrato no está validado.", f"Corre `python scripts/contract.py validate --run {run}`.")], []
    return [], list(rec["hashes"].keys())


def init(run, tipo: str | None) -> None:
    run = Path(run)
    run.mkdir(parents=True, exist_ok=True)
    (run / "data").mkdir(exist_ok=True)
    (run / "experiments").mkdir(exist_ok=True)
    dest = run / "contract.yaml"
    if dest.exists():
        raise HarnessError(f"ya existe {dest}.", "Edítalo directamente; init no sobrescribe.")
    txt = (TEMPLATES / "contract.yaml").read_text(encoding="utf-8")
    if tipo:
        txt = txt.replace("problem_type: POR_DEFINIR", f"problem_type: {tipo}", 1)
    dest.write_text(txt, encoding="utf-8")
    dec = run / "decisions.md"
    if not dec.exists():
        dec.write_text((TEMPLATES / "decisions.md").read_text(encoding="utf-8"), encoding="utf-8")
    shutil.copy2(TEMPLATES / "experiment_template.py", run / "experiments" / "_plantilla.py")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("accion", choices=["init", "check", "validate"])
    ap.add_argument("--run", required=True)
    ap.add_argument("--type", choices=TIPOS)
    ap.add_argument("--new-version", action="store_true", help="re-validar después de observar el holdout")
    a = ap.parse_args()
    if a.accion == "init":
        init(a.run, a.type)
        print(f"OK: run creado en {a.run}. Completa contract.yaml (todo POR_DEFINIR) y decisions.md.")
        return 0
    if a.accion == "check":
        c = cargar_contrato(a.run)
        cols, clases = None, None
        ruta = _g(c, "datos.ruta")
        if ruta and ruta != PD and (Path(a.run) / ruta).exists():
            raw = read_table(Path(a.run) / ruta)
            cols = list(raw.columns)
            if c.get("problem_type") == "classification" and _g(c, "datos.target") in raw:
                clases = raw[c["datos"]["target"]].nunique()
        probs = revisar(c, cols, clases)
        if probs:
            raise HarnessError("contrato incompleto o incoherente:\n" + "\n".join(f"  - {q}" for q, _ in probs),
                               "\n" + "\n".join(f"  - {w}" for _, w in probs))
        print("OK: contrato completo y coherente" + ("" if cols else " (sin revisar columnas: no hay snapshot)"))
        return 0
    r = validar(a.run, a.new_version)
    print(f"OK: contrato validado. dev={r['dev']} filas, holdout={r['holdout']} filas sellado, {r['folds']} folds, "
          f"duplicados eliminados={r['duplicados_eliminados']}.")
    return 0


if __name__ == "__main__":
    cli(main)
