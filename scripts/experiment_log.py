"""
Fase 3 — Runner y bitácora de experimentos (`experiments.jsonl`, solo se agrega).

Las métricas las calcula el harness, no el agente. Cada experimento es un archivo
`experiments/E0xx.py` con HIPOTESIS, FEATURES, FAMILIA, ESTOCASTICO, CONFIG y
`build_pipeline(config, seed)` (ver experiments/_plantilla.py).

Uso:
    python scripts/experiment_log.py run E003 --run runs/<p> [--repetir]
    python scripts/experiment_log.py conclude E003 --resultado "..." --decision aceptada|descartada|inconclusa --run runs/<p>
    python scripts/experiment_log.py stop --motivo saturacion|presupuesto|sin_alternativas --evidencia "..." --run runs/<p>
    python scripts/experiment_log.py list --run runs/<p>

`run`:
  * usa los folds congelados de splits.json; ajusta un pipeline NUEVO por fold y semilla solo con
    el entrenamiento del fold (todo lo aprendido — imputación, escalado, codificación, selección,
    remuestreo, umbral — queda dentro del fold);
  * corre todas las semillas del contrato si ESTOCASTICO=True, y una sola si no;
  * valida alineación, cobertura y finitud de las predicciones;
  * guarda predicciones fuera de muestra en preds/<run_id> (id de fila, fold, semilla, origen, horizonte);
  * registra métricas por fold y semilla, tiempos, hashes, copia exacta del código ejecutado
    (experiments/_executed/) y versiones del entorno.
Los experimentos fallidos se registran con su error; no desaparecen del historial.
Un experimento ya ejecutado (y vigente) no se sobrescribe: si cambias el código, crea otro id.
"""
from __future__ import annotations

import argparse
import importlib.util
import re
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from common import (HarnessError, append_jsonl, atomic_write_text, cli, env_info, ahora, read_json, read_jsonl, sha256_file,
                    sha256_json, sha256_text, write_table)
from metrics import METRICAS, calcular

CATEGORIAS = ("features", "familia_modelo", "desbalance_outliers", "transformacion_target", "regularizacion",
              "hiperparametros", "otra")
DECISIONES = ("aceptada", "descartada", "inconclusa")
MOTIVOS_CIERRE = ("saturacion", "presupuesto", "sin_alternativas")


# ------------------------------------------------------------------ registros

def bitacora(run) -> list[dict]:
    return read_jsonl(Path(run) / "experiments.jsonl")


def registros_vigentes(run, ctx: dict | None = None, incluir_fallidos=False) -> list[dict]:
    """Ejecuciones (baseline/experimento) con el eval_hash y splits_hash vigentes."""
    from contract import contexto_eval
    ctx = ctx or contexto_eval(run)
    out = []
    for r in bitacora(run):
        if r.get("tipo") not in ("baseline", "experimento"):
            continue
        if r["eval_hash"] != ctx["eval_hash"] or r["splits_hash"] != ctx["splits_hash"]:
            continue
        if r["estado"] != "ok" and not incluir_fallidos:
            continue
        out.append(r)
    return out


def normalizar(txt: str) -> str:
    return re.sub(r"[^a-z0-9áéíóúñ]+", " ", (txt or "").lower()).strip()


# ------------------------------------------------------------------ carga del experimento

def cargar_modulo(path: Path, nombre: str):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    spec = importlib.util.spec_from_file_location(f"exp_{nombre}_{abs(hash(str(path)))}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def spec_experimento(run, exp_id: str) -> dict:
    p = Path(run) / "experiments" / f"{exp_id}.py"
    if not re.fullmatch(r"E\d{3}", exp_id):
        raise HarnessError(f"id de experimento inválido: {exp_id}", "Usa E001, E002, ... (archivo experiments/E0xx.py).")
    if not p.exists():
        raise HarnessError(f"no existe {p}.", "Copia experiments/_plantilla.py a ese nombre y llénalo.")
    codigo = p.read_text(encoding="utf-8")
    mod = cargar_modulo(p, exp_id)
    falt = [k for k in ("HIPOTESIS", "FEATURES", "FAMILIA", "ESTOCASTICO", "CONFIG", "build_pipeline") if not hasattr(mod, k)]
    if falt:
        raise HarnessError(f"{p.name} no define {falt}.", "Sigue la interfaz de experiments/_plantilla.py.")
    H = mod.HIPOTESIS
    faltan = [k for k in ("hipotesis", "categoria", "referencia", "cambio", "esperado", "criterio_descarte") if not str(H.get(k, "")).strip()]
    if faltan:
        raise HarnessError(f"la HIPOTESIS de {exp_id} no declara {faltan}.",
                           "Cada experimento declara hipótesis, categoría, referencia, cambio, resultado esperado y criterio de descarte.")
    if H["categoria"] not in CATEGORIAS:
        raise HarnessError(f"categoría inválida: {H['categoria']!r}.", f"Usa una de {CATEGORIAS}.")
    return {"id": exp_id, "tipo": "experimento", "hipotesis": dict(H), "features": list(mod.FEATURES),
            "familia": mod.FAMILIA, "estocastico": bool(mod.ESTOCASTICO), "config": dict(mod.CONFIG),
            "build": mod.build_pipeline, "codigo": codigo, "codigo_ruta": p.relative_to(run).as_posix()}


def spec_desde_registro(run, rec: dict) -> dict:
    """Reconstruye el spec EXACTO de una ejecución registrada (código congelado en _executed/ para
    experimentos; definición de baselines.py para baselines). Falla si el código ya no coincide con su hash."""
    run = Path(run)
    if rec["tipo"] == "experimento":
        p = run / rec["codigo_ref"]
        if not p.exists():
            raise HarnessError(f"no existe el código congelado {rec['codigo_ref']}.", "Restaura experiments/_executed/ desde git o vuelve a correr el experimento con un id nuevo.")
        codigo = p.read_text(encoding="utf-8")
        if sha256_text(codigo) != rec["codigo_sha256"]:
            raise HarnessError(f"el código congelado de {rec['run_id']} no coincide con su hash.", "Fue modificado después de ejecutarse; vuelve a correr el experimento con un id nuevo.")
        mod = cargar_modulo(p, rec["run_id"].replace("#", "_"))
        return {"id": rec["id"], "tipo": "experimento", "features": list(rec["features"]), "familia": rec["familia"],
                "estocastico": rec["estocastico"], "config": dict(rec["config"]), "build": mod.build_pipeline, "codigo": codigo}
    import inspect
    from baselines import FUENTE, escalera
    from contract import cargar_contrato
    b = [x for x in escalera(cargar_contrato(run), rec["features"] or ["_"]) if x["id"] == rec["id"]]
    if not b:
        raise HarnessError(f"el baseline {rec['id']} ya no existe en baselines.py.", "Vuelve a correr baselines.py.")
    b = b[0]
    codigo = FUENTE + f"\n# baseline {b['id']} ({b['nombre']}): " + inspect.getsource(b["build"]).strip()
    if sha256_text(codigo) != rec["codigo_sha256"]:
        raise HarnessError(f"baselines.py cambió desde que se ejecutó {rec['run_id']}.", "Vuelve a correr baselines.py y la comparación.")
    return {"id": rec["id"], "tipo": "baseline", "features": list(rec["features"]), "familia": rec["familia"],
            "estocastico": False, "config": {}, "build": b["build"], "codigo": codigo, "contexto_ok": True}


# ------------------------------------------------------------------ ejecución

def _es_ajustado(est) -> bool:
    from sklearn.exceptions import NotFittedError
    from sklearn.utils.validation import check_is_fitted
    try:
        check_is_fitted(est)
        return True
    except (NotFittedError, TypeError):
        return False


def _requiere_contexto(est) -> bool:
    if getattr(est, "requiere_contexto", False):
        return True
    steps = getattr(est, "steps", None)
    return bool(steps) and getattr(steps[-1][1], "requiere_contexto", False)


def _proba(est, X, clases) -> np.ndarray:
    if not hasattr(est, "predict_proba"):
        raise ValueError("el modelo no implementa predict_proba y la métrica/umbral lo requiere")
    P = est.predict_proba(X)
    cl = list(getattr(est, "classes_", clases))
    out = np.zeros((len(X), len(clases)))
    for j, k in enumerate(cl):
        out[:, clases.index(k)] = P[:, j]
    return out


def _umbral_interno(build, config, seed, Xtr, ytr, clases, objetivo, grupos=None) -> float:
    """Ajusta el umbral DENTRO del entrenamiento del fold con una partición interna."""
    from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit
    sp = (GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed) if grupos is not None
          else StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=seed))
    a, b = next(sp.split(Xtr, ytr, groups=grupos))
    m = build(config, seed)
    m.fit(Xtr.iloc[a], ytr.iloc[a])
    p = _proba(m, Xtr.iloc[b], clases)[:, 1]
    yb = ytr.iloc[b].values
    mejor, mejor_u = -1, 0.5
    for u in np.unique(np.quantile(p, np.linspace(0.02, 0.98, 49))):
        yh = np.where(p >= u, clases[1], clases[0])
        s = calcular(objetivo, yb, yh, None, clases)
        if s > mejor:
            mejor, mejor_u = s, float(u)
    return mejor_u


def ajustar_predecir(spec: dict, c: dict, tr: pd.DataFrame, te: pd.DataFrame, s: int, clases, umbral) -> dict:
    """Construye el pipeline con la semilla `s`, lo ajusta SOLO con `tr` y predice `te`.
    Lo usan el CV, las ablaciones, la auditoría y el holdout (mismo procedimiento en todos)."""
    target = c["datos"]["target"]
    clf = clases is not None
    metricas = [c["metrica"]["principal"], *(c["metrica"].get("secundarias") or [])]
    est = spec["build"](spec["config"], s)
    if _es_ajustado(est):
        raise RuntimeError("build_pipeline devolvió un estimador YA AJUSTADO: se habría ajustado fuera del fold.")
    cols = spec["features"] + ([k for k in ("_entidad", "_t") if k in tr] if _requiere_contexto(est) else [])
    Xtr, ytr, Xte = tr[cols], tr[target], te[cols]
    u = None
    if umbral and umbral.get("politica") == "ajustada_interna":
        g = tr["_entidad"].values if "_entidad" in tr and c["validacion"]["esquema"] == "grupos" else None
        u = _umbral_interno(spec["build"], spec["config"], s, Xtr, ytr, clases, umbral.get("objetivo", "f1"), g)
    elif umbral:
        u = float(umbral.get("valor", 0.5))
    a = time.time()
    est.fit(Xtr, ytr)
    t_fit = time.time() - a
    a = time.time()
    if clf:
        P = _proba(est, Xte, clases) if (umbral or any(METRICAS[k].requiere_proba for k in metricas)) else None
        if P is not None and len(clases) == 2:
            yhat = np.where(P[:, 1] >= u, clases[1], clases[0])
        elif P is not None:
            yhat = np.array(clases)[P.argmax(1)]
        else:
            yhat = np.asarray(est.predict(Xte))
    else:
        P, yhat = None, np.asarray(est.predict(Xte), float)
    t_pred = time.time() - a
    if len(yhat) != len(te):
        raise RuntimeError(f"{len(yhat)} predicciones para {len(te)} filas (desalineadas).")
    if not clf and not np.all(np.isfinite(yhat)):
        raise RuntimeError(f"predicciones no finitas o faltantes ({int((~np.isfinite(yhat)).sum())}).")
    if P is not None and (not np.all(np.isfinite(P)) or np.abs(P.sum(1) - 1).max() > 1e-6):
        raise RuntimeError("probabilidades inválidas (no finitas o no suman 1).")
    return {"yhat": yhat, "P": P, "umbral": u, "t_fit": t_fit, "t_pred": t_pred, "estimador": est}


def contexto_cv(run) -> dict:
    """Contrato, datos de desarrollo, splits y parámetros derivados que comparte todo el CV."""
    from contract import cargar_contrato, cargar_dev
    run = Path(run)
    c = cargar_contrato(run)
    dev = cargar_dev(run)
    m = c["metrica"]
    clf = c["problem_type"] == "classification"
    clases = sorted(dev[c["datos"]["target"]].unique().tolist()) if clf else None
    return {"c": c, "dev": dev, "sp": read_json(run / "splits.json"), "clases": clases,
            "metricas": [m["principal"], *(m.get("secundarias") or [])],
            "umbral": (m.get("umbral") or {"politica": "fija", "valor": 0.5}) if clf and len(clases) == 2 else None,
            "pos": pd.Series(np.arange(len(dev)), index=dev[c["datos"]["id_fila"]].astype(str))}


def correr_cv(K: dict, spec: dict, semillas: list[int], folds: list[dict] | None = None) -> dict:
    """CV con los folds congelados. Devuelve métricas por fold/semilla, predicciones OOF, tiempos y umbrales.
    Lanza excepción si algo falla (el llamador decide si registrarlo)."""
    c, dev, clases, metricas = K["c"], K["dev"], K["clases"], K["metricas"]
    idc, target = c["datos"]["id_fila"], c["datos"]["target"]
    folds = folds if folds is not None else K["sp"]["folds"]
    filas, preds, umbrales, fit_s, pred_s, n_pred = [], [], [], [], 0.0, 0
    for fo in folds:
        tr, va = dev.iloc[K["pos"][fo["train"]].values], dev.iloc[K["pos"][fo["val"]].values]
        for s in semillas:
            try:
                r = ajustar_predecir(spec, c, tr, va, s, clases, K["umbral"])
            except Exception as e:
                raise RuntimeError(f"fold {fo['id']}, semilla {s}: {type(e).__name__}: {e}") from e
            fit_s.append(r["t_fit"])
            pred_s += r["t_pred"]
            n_pred += len(va)
            if r["umbral"] is not None:
                umbrales.append({"fold": fo["id"], "semilla": s, "umbral": r["umbral"]})
            fila = {"fold": fo["id"], "semilla": s}
            for k in metricas:
                fila[k] = calcular(k, va[target].values, r["yhat"], r["P"], clases)
            filas.append(fila)
            pf = pd.DataFrame({"id": va[idc].astype(str).values, "fold": fo["id"], "repeticion": fo["repeticion"],
                               "semilla": s, "y_true": va[target].values, "y_pred": r["yhat"]})
            if r["P"] is not None:
                for j, k in enumerate(clases):
                    pf[f"p__{k}"] = r["P"][:, j]
            if "_entidad" in dev:
                pf["entidad"] = va["_entidad"].values
            if fo.get("origen") is not None:
                pf["t"] = va["_t"].values
                pf["origen"] = fo["origen"]
                pf["horizonte"] = pf["t"] - fo["origen"]
            preds.append(pf)
    preds = pd.concat(preds, ignore_index=True)
    esperados = {i for fo in folds for i in fo["val"]}
    if set(preds["id"]) != esperados:
        raise RuntimeError(f"cobertura incompleta: {len(esperados - set(preds['id']))} filas de validación sin predicción.")
    return {"filas": filas, "preds": preds, "umbrales": umbrales,
            "tiempos": {"fit_s_total": round(sum(fit_s), 4), "fit_s_max_fold": round(max(fit_s), 4) if fit_s else None,
                        "pred_ms_1k": round(1e6 * pred_s / max(n_pred, 1), 4)}}


def ejecutar(run, spec: dict, repetir_de: str | None = None) -> dict:
    """Corre `spec` con los folds congelados y agrega el registro a experiments.jsonl."""
    from contract import contexto_eval, estado_feature, features_utilizables
    run = Path(run)
    K = contexto_cv(run)
    c, dev, sp, clases = K["c"], K["dev"], K["sp"], K["clases"]
    ctx = contexto_eval(run)
    usables = set(features_utilizables(c, dev.columns))
    malas = {f: estado_feature(c, f)[1] for f in spec["features"] if f not in usables}
    if malas:
        raise HarnessError(f"{spec['id']} usa features no permitidas por el contrato: {malas}.",
                           "Quítalas del experimento o corrige su disponibilidad en el contrato (con su decisión en decisions.md).")
    if not spec["features"] and not spec.get("contexto_ok"):
        raise HarnessError(f"{spec['id']} no declara features.", "Declara FEATURES.")
    semillas = list(c.get("experimentos", {}).get("semillas") or [0])
    semillas = semillas if spec["estocastico"] else semillas[:1]
    t0 = time.time()
    n_prev = sum(1 for r in bitacora(run) if r.get("id") == spec["id"] and r.get("tipo") in ("baseline", "experimento"))
    run_id = spec["id"] if n_prev == 0 else f"{spec['id']}#{n_prev + 1}"
    codigo_sha = sha256_text(spec["codigo"])
    rec = {"tipo": spec["tipo"], "id": spec["id"], "run_id": run_id, "fecha": ahora(), "estado": "ok", "error": None,
           "hipotesis": spec.get("hipotesis"), "nombre": spec.get("nombre"), "features": spec["features"],
           "familia": spec["familia"], "estocastico": spec["estocastico"], "config": spec["config"], "semillas": semillas,
           "eval_hash": ctx["eval_hash"], "splits_hash": ctx["splits_hash"], "version_estudio": ctx["version"],
           "codigo_sha256": codigo_sha, "firma_config": sha256_json([sorted(spec["features"]), spec["familia"], spec["config"], codigo_sha]),
           "reproduccion_de": repetir_de, "entorno": env_info(), "metricas_por_fold": [], "umbrales": [], "tiempos": {}}
    exe = run / "experiments" / "_executed" / f"{run_id.replace('#', '__')}.py"
    atomic_write_text(exe, spec["codigo"])
    rec["codigo_ref"] = exe.relative_to(run).as_posix()
    try:
        R = correr_cv(K, spec, semillas)
        rec["metricas_por_fold"], rec["umbrales"], rec["tiempos"] = R["filas"], R["umbrales"], R["tiempos"]
        p = write_table(R["preds"], run / "preds" / run_id.replace("#", "__"))
        rec["preds_ref"] = p.relative_to(run).as_posix()
        rec["preds_sha256"] = sha256_file(p)
        rec["resumen"] = resumir(R["filas"], sp, K["metricas"], c["metrica"].get("agregacion", "media_folds"), R["preds"], clases)
    except Exception as e:  # el fallo queda registrado, no desaparece
        rec["estado"] = "fallido"
        rec["error"] = f"{type(e).__name__}: {e}"
        rec["traceback"] = traceback.format_exc()[-3000:]
    rec["tiempos"]["duracion_s"] = round(time.time() - t0, 2)
    rec["clases"] = clases
    append_jsonl(run, run / "experiments.jsonl", rec)
    return rec


def agregar(df_fold: pd.DataFrame, k: str) -> float:
    """Media sobre folds dentro de cada repetición y luego sobre repeticiones."""
    return float(df_fold.groupby("repeticion")[k].mean().mean())


def resumir(filas: list[dict], sp: dict, metricas, agregacion: str, preds: pd.DataFrame, clases) -> dict:
    rep = {f["id"]: f["repeticion"] for f in sp["folds"]}
    df = pd.DataFrame(filas)
    df["repeticion"] = df["fold"].map(rep)
    out = {}
    for k in metricas:
        por_semilla = {}
        for s, g in df.groupby("semilla"):
            if agregacion == "pooled":
                ps = preds[preds["semilla"] == s]
                vals = []
                for r, gp in ps.groupby("repeticion"):
                    P = gp[[f"p__{c}" for c in clases]].values if clases and f"p__{clases[0]}" in gp else None
                    vals.append(calcular(k, gp["y_true"].values, gp["y_pred"].values, P, clases))
                por_semilla[int(s)] = float(np.mean(vals))
            else:
                por_semilla[int(s)] = agregar(g, k)
        media_fold = df.groupby("fold")[k].mean()  # promedio sobre semillas por fold
        out[k] = {"valor": float(np.mean(list(por_semilla.values()))),
                  "std_entre_folds": float(media_fold.std(ddof=1)) if len(media_fold) > 1 else None,
                  "min_fold": float(media_fold.min()), "max_fold": float(media_fold.max()),
                  "por_semilla": por_semilla,
                  "std_entre_semillas": float(np.std(list(por_semilla.values()), ddof=1)) if len(por_semilla) > 1 else None}
    return out


# ------------------------------------------------------------------ comandos

def cmd_run(run, exp_id: str, repetir: bool) -> int:
    from contract import cargar_contrato, contexto_eval
    from gate import requerir
    requerir(run, "experimentos")
    spec = spec_experimento(run, exp_id)
    ctx = contexto_eval(run)
    previos = [r for r in registros_vigentes(run, ctx) if r["id"] == exp_id]
    repetir_de = None
    if previos:
        ult = previos[-1]
        if ult["codigo_sha256"] != sha256_text(spec["codigo"]):
            raise HarnessError(f"{exp_id} ya se ejecutó ({ult['run_id']}) y su código cambió después.",
                               "Los experimentos ejecutados no se sobrescriben: crea un id nuevo (E0xx siguiente) con el cambio.")
        if not repetir:
            raise HarnessError(f"{exp_id} ya se ejecutó con este código ({ult['run_id']}).",
                               "Usa sus resultados. Para medir reproducibilidad usa --repetir (no cuenta como hipótesis nueva).")
        repetir_de = ult["run_id"]
    c = cargar_contrato(run)
    pres = int((c.get("experimentos") or {}).get("presupuesto", 30))
    usados = len({r["id"] for r in registros_vigentes(run, ctx, incluir_fallidos=True) if r["tipo"] == "experimento"})
    if not previos and usados >= pres:
        raise HarnessError(f"presupuesto de experimentos agotado ({usados}/{pres}).",
                           "Cierra la fase con `experiment_log.py stop --motivo presupuesto ...` o cambia el presupuesto en el contrato con su decisión en decisions.md.")
    for r in registros_vigentes(run, ctx):
        if r["tipo"] == "experimento" and r["id"] != exp_id and normalizar(r["hipotesis"]["hipotesis"]) == normalizar(spec["hipotesis"]["hipotesis"]):
            print(f"AVISO: la hipótesis de {exp_id} repite textualmente la de {r['id']}; no contará como hipótesis distinta.")
    rec = ejecutar(run, spec, repetir_de)
    if rec["estado"] != "ok":
        print(f"FALLÓ {rec['run_id']} (registrado en experiments.jsonl): {rec['error']}")
        return 1
    pm = c["metrica"]["principal"]
    r = rec["resumen"][pm]
    print(f"OK {rec['run_id']}: {pm} = {r['valor']:.5g} (std entre folds {r['std_entre_folds']:.3g}"
          + (f", std entre semillas {r['std_entre_semillas']:.3g}" if r["std_entre_semillas"] is not None else "")
          + f"). Registra el resultado con `experiment_log.py conclude {exp_id} ...`.")
    return 0


def cmd_conclude(run, exp_id, resultado, decision) -> int:
    from contract import contexto_eval
    if decision not in DECISIONES:
        raise HarnessError(f"decisión inválida: {decision}", f"Usa {DECISIONES}.")
    if len((resultado or "").strip()) < 30:
        raise HarnessError("el resultado observado es demasiado corto.", "Describe qué se observó vs lo esperado (>= 30 caracteres), con cifras.")
    regs = [r for r in registros_vigentes(run, contexto_eval(run), incluir_fallidos=True) if r["id"] == exp_id]
    if not regs:
        raise HarnessError(f"{exp_id} no tiene ejecuciones vigentes.", f"Córrelo primero con `experiment_log.py run {exp_id}`.")
    append_jsonl(run, Path(run) / "experiments.jsonl", {"tipo": "conclusion", "id": exp_id, "run_id": regs[-1]["run_id"],
                                                        "resultado_observado": resultado, "decision": decision, "fecha": ahora()})
    print(f"OK: conclusión de {exp_id} registrada ({decision}).")
    return 0


def cmd_stop(run, motivo, evidencia) -> int:
    if motivo not in MOTIVOS_CIERRE:
        raise HarnessError(f"motivo inválido: {motivo}", f"Usa {MOTIVOS_CIERRE}.")
    if len((evidencia or "").strip()) < 40:
        raise HarnessError("la evidencia del cierre anticipado es insuficiente.",
                           "Explica con cifras por qué no hay alternativas útiles (>= 40 caracteres). El revisor la evaluará.")
    from contract import contexto_eval
    append_jsonl(run, Path(run) / "experiments.jsonl", {"tipo": "cierre", "motivo": motivo, "evidencia": evidencia,
                                                        "eval_hash": contexto_eval(run)["eval_hash"], "fecha": ahora()})
    print("OK: cierre anticipado registrado (sujeto a revisión).")
    return 0


def cmd_list(run) -> int:
    from contract import cargar_contrato, contexto_eval
    pm = cargar_contrato(run)["metrica"]["principal"]
    ctx = contexto_eval(run)
    concl = {r["id"]: r for r in bitacora(run) if r.get("tipo") == "conclusion"}
    for r in bitacora(run):
        if r.get("tipo") not in ("baseline", "experimento"):
            continue
        vig = "vigente" if (r["eval_hash"], r["splits_hash"]) == (ctx["eval_hash"], ctx["splits_hash"]) else "obsoleto"
        val = r.get("resumen", {}).get(pm, {}).get("valor") if r["estado"] == "ok" else None
        txt = f"{val:.5g}" if val is not None else r.get("error", "")[:60]
        print(f"{r['run_id']:<8} {r['tipo']:<11} {r['estado']:<8} {vig:<9} {r['familia']:<8} {pm}={txt}  "
              f"{concl.get(r['id'], {}).get('decision', '')}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("accion", choices=["run", "conclude", "stop", "list"])
    ap.add_argument("exp_id", nargs="?")
    ap.add_argument("--run", required=True)
    ap.add_argument("--repetir", action="store_true")
    ap.add_argument("--resultado")
    ap.add_argument("--decision")
    ap.add_argument("--motivo")
    ap.add_argument("--evidencia")
    a = ap.parse_args()
    run = Path(a.run)
    if a.accion in ("run", "conclude") and not a.exp_id:
        raise HarnessError("falta el id del experimento.", "Ejemplo: experiment_log.py run E001 --run ...")
    if a.accion == "run":
        return cmd_run(run, a.exp_id, a.repetir)
    if a.accion == "conclude":
        return cmd_conclude(run, a.exp_id, a.resultado, a.decision)
    if a.accion == "stop":
        return cmd_stop(run, a.motivo, a.evidencia)
    return cmd_list(run)


if __name__ == "__main__":
    cli(main)
