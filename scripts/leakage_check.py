"""
Fase 1 — Revisión de leakage por evidencia y severidad.

Uso:
    python scripts/leakage_check.py --run runs/<p>

Separa dos clases de hallazgos:

VIOLACIONES COMPROBADAS (severidad bloqueante; no se pueden "justificar", solo corregir):
  V-DISP-<f>     feature declarada `posterior`, o `con_retraso` con retraso < horizonte, o con
                 linaje que dice que se deriva del target sin desfase suficiente.
  V-IDENT-<f>    la feature ES el target (igual o transformación afín exacta).
  V-DESFASE-<f>  la feature coincide exactamente con el target desplazado k periodos dentro de cada
                 entidad, con k < horizonte (k<0 = información futura).
  V-SPLIT-n      los splits materializados violan el contrato (solapamiento, gap, entidades).
  V-DUP          duplicados accidentales que cruzan entrenamiento/validación.

SEÑALES SOSPECHOSAS (requieren investigación; se cierran excluyendo la feature o justificándola):
  S-UNIVAR-<f>       poder univariado casi perfecto MEDIDO FUERA DE MUESTRA (R² o AUC con los folds del contrato).
  S-CORR-<f>         |Spearman| con el target >= umbral.
  S-TEMPORAL-<f>     la feature se asocia más con el target FUTURO que con el presente/pasado.
  S-IMPORTANCIA-<f>  una feature concentra casi toda la importancia por permutación.
  S-METADATA-<f>     disponibilidad `desconocida` (con excepción documentada en el contrato).
  S-DUP              duplicados declarados legítimos que cruzan entrenamiento/validación.

Correlación o importancia altas NO prueban leakage, y un rezago legítimo del target o una
variable futura conocida (calendario) no se bloquean: por eso son señales, no violaciones.

Cómo se resuelve:
  * Excluir la feature: agrégala a `datos.columnas_excluidas` en contract.yaml, registra la
    decisión en decisions.md, vuelve a correr contract.py validate y este script.
  * Justificar una SEÑAL: en `leakage_resolutions.yaml`:
        S-CORR-ventas_lag7:
          estado: justificada
          justificacion: "rezago de 7 días, retraso 7 = horizonte; la correlación viene de la persistencia de la serie"
    Una justificación NO convierte una violación comprobada en evaluación válida.

Escribe leakage_report.json, leakage_report.md y el bloque de leakage de eda_report.md.
Sale con código 1 si el veredicto es `bloqueado`.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from common import (HarnessError, atomic_write_json, atomic_write_text, cli, find_table, read_json, sha256_file,
                    sha256_json)

MIN_JUST = 30


# ------------------------------------------------------------------ utilidades compartidas con audit.py

def matriz_numerica(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Codifica categóricas como enteros (solo para chequeos de diagnóstico, no para modelos)."""
    X = pd.DataFrame(index=df.index)
    for k in cols:
        s = df[k]
        X[k] = s.astype("category").cat.codes.replace(-1, np.nan) if not pd.api.types.is_numeric_dtype(s) else s.astype(float)
    return X


def desplazar(df: pd.DataFrame, col: str, k: int) -> pd.Series:
    """col_{t-k} dentro de cada entidad (k>0 pasado, k<0 futuro). Requiere `_t`."""
    orden = df.sort_values(["_entidad", "_t"] if "_entidad" in df else ["_t"])
    if "_entidad" in df:
        s = orden.groupby("_entidad")[col].shift(k)
    else:
        s = orden[col].shift(k)
    return s.reindex(df.index)


def coincide(a: pd.Series, b: pd.Series, frac=0.999) -> bool:
    m = a.notna() & b.notna()
    if m.sum() < max(10, 0.5 * len(a)):
        return False
    x, y = a[m].astype(float).values, b[m].astype(float).values
    escala = max(np.abs(y).mean(), 1e-9)
    return float(np.mean(np.isclose(x, y, rtol=1e-6, atol=1e-6 * escala))) >= frac


def desfase_exacto(df: pd.DataFrame, f: str, target: str, kmax: int) -> int | None:
    """Devuelve k si f_t == target_{t-k} (dentro de entidad) para algún |k| <= kmax."""
    if "_t" not in df or not pd.api.types.is_numeric_dtype(df[f]):
        return None
    for k in sorted(range(-kmax, kmax + 1), key=abs):
        if k == 0:
            continue
        if coincide(df[f], desplazar(df, target, k)):
            return k
    return None


def perfil_temporal(df: pd.DataFrame, f: str, target: str, kmax: int) -> dict | None:
    """Correlación (demeaned por entidad) de f_t con target_{t+k} para k en [-kmax, kmax]."""
    if "_t" not in df or not pd.api.types.is_numeric_dtype(df[f]):
        return None
    base = df[[f, target]].copy()
    if "_entidad" in df:
        g = df["_entidad"]
        base[f] = base[f] - base.groupby(g)[f].transform("mean")
    out = {}
    for k in range(-kmax, kmax + 1):
        y = desplazar(df, target, -k)  # target en t+k
        if "_entidad" in df:
            y = y - y.groupby(df["_entidad"]).transform("mean")
        m = base[f].notna() & y.notna()
        if m.sum() > 20 and base.loc[m, f].std() > 0 and y[m].std() > 0:
            out[k] = float(np.corrcoef(base.loc[m, f], y[m])[0, 1])
    if not out:
        return None
    futuro = {k: v for k, v in out.items() if k > 0}
    resto = {k: v for k, v in out.items() if k <= 0}
    kf = max(futuro, key=lambda k: abs(futuro[k])) if futuro else None
    return {"corr_por_desfase": out, "max_futuro": abs(futuro[kf]) if kf else 0.0, "k_futuro": kf,
            "max_presente_pasado": max(abs(v) for v in resto.values()) if resto else 0.0}


def folds_dev(run, n_max=3) -> list[dict]:
    sp = read_json(Path(run) / "splits.json")
    fs = [f for f in sp["folds"] if f["repeticion"] == 0]
    return fs[-n_max:] if sp["esquema"] == "temporal" else fs[:n_max]


def univariado_oof(df: pd.DataFrame, f: str, c: dict, folds: list[dict]) -> float | None:
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    from sklearn.metrics import r2_score, roc_auc_score
    idc, target = c["datos"]["id_fila"], c["datos"]["target"]
    X = matriz_numerica(df, [f])
    y = df[target]
    pos = pd.Series(np.arange(len(df)), index=df[idc].astype(str))
    pred, real = [], []
    clf = c["problem_type"] == "classification"
    for fo in folds:
        tr, va = pos[fo["train"]].values, pos[fo["val"]].values
        if clf and y.iloc[tr].nunique() < 2:
            continue
        m = (HistGradientBoostingClassifier if clf else HistGradientBoostingRegressor)(max_iter=60, max_depth=3, random_state=0)
        m.fit(X.iloc[tr], y.iloc[tr])
        if clf:
            P = m.predict_proba(X.iloc[va])
            pred.append(P[:, 1] if P.shape[1] == 2 else P)
        else:
            pred.append(m.predict(X.iloc[va]))
        real.append(y.iloc[va].values)
    if not pred:
        return None
    yt = np.concatenate(real)
    yp = np.concatenate(pred)
    if clf:
        if len(np.unique(yt)) < 2:
            return None
        return float(roc_auc_score(yt, yp, multi_class="ovr") if yp.ndim == 2 else roc_auc_score(yt, yp))
    return float(r2_score(yt, yp))


def dominancia_permutacion(df: pd.DataFrame, feats: list[str], c: dict, fold: dict) -> dict | None:
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    from sklearn.inspection import permutation_importance
    if len(feats) < 3:
        return None
    idc, target = c["datos"]["id_fila"], c["datos"]["target"]
    X = matriz_numerica(df, feats)
    y = df[target]
    pos = pd.Series(np.arange(len(df)), index=df[idc].astype(str))
    tr, va = pos[fold["train"]].values, pos[fold["val"]].values
    clf = c["problem_type"] == "classification"
    m = (HistGradientBoostingClassifier if clf else HistGradientBoostingRegressor)(max_iter=100, random_state=0)
    m.fit(X.iloc[tr], y.iloc[tr])
    r = permutation_importance(m, X.iloc[va], y.iloc[va], n_repeats=3, random_state=0,
                               scoring="neg_log_loss" if clf else "r2")
    imp = pd.Series(np.clip(r.importances_mean, 0, None), index=feats).sort_values(ascending=False)
    tot = imp.sum()
    if tot <= 0:
        return None
    return {"top": imp.index[0], "participacion": float(imp.iloc[0] / tot), "importancias": imp.round(5).to_dict()}


# ------------------------------------------------------------------ revisión

def _h(id_, clase, tipo, feature, evidencia, accion):
    return {"id": id_, "clase": clase, "tipo": tipo, "feature": feature,
            "severidad": "bloqueante" if clase == "violacion" else "alta",
            "estado": "abierto", "evidencia": evidencia, "accion": accion, "justificacion": None}


def revisar(run) -> dict:
    from contract import cargar_contrato, cargar_dev, columnas_estructurales, estado_feature, COLS_HARNESS
    from cv import validar_splits
    run = Path(run)
    c = cargar_contrato(run)
    df = cargar_dev(run)
    d, lk = c["datos"], (c.get("leakage") or {})
    target = d["target"]
    h = int(c["prediccion"].get("horizonte", 0) or 0)
    est = int(c["prediccion"].get("estacionalidad") or 0)
    kmax = min(max(h, est, 7), 28)
    estr = columnas_estructurales(c)
    candidatas = [k for k in df.columns if k not in estr and k not in COLS_HARNESS]
    H: list[dict] = []
    revisadas, excluidas = [], []
    excl_como = ("Agrega la columna a datos.columnas_excluidas en contract.yaml (o corrige su disponibilidad si estaba mal "
                 "declarada), registra la decisión en decisions.md y vuelve a correr contract.py validate y leakage_check.py.")

    for f in candidatas:
        e, motivo = estado_feature(c, f)
        if e == "excluida":
            excluidas.append(f)
            continue
        if e == "violacion":
            H.append(_h(f"V-DISP-{f}", "violacion", "disponibilidad", f, motivo, excl_como))
            continue
        if e == "sin_declarar":
            H.append(_h(f"V-DISP-{f}", "violacion", "disponibilidad", f, motivo, "Declara su disponibilidad en el contrato."))
            continue
        revisadas.append(f)
        if e == "excepcion":
            s = _h(f"S-METADATA-{f}", "senal", "metadata_incompleta", f,
                   "disponibilidad desconocida; aceptada por excepción en el contrato", "Resolver la disponibilidad cuando sea posible.")
            s.update(severidad="media", estado="justificada", justificacion=c["excepciones_disponibilidad"][f])
            H.append(s)

    # identidad y desfases exactos (pruebas deterministas)
    for f in revisadas:
        if not pd.api.types.is_numeric_dtype(df[f]):
            continue
        if coincide(df[f], df[target].astype(float)) or (
                df[f].notna().sum() > 10 and df[f].std() > 0 and abs(np.corrcoef(df[f].fillna(df[f].mean()), df[target].astype(float))[0, 1]) > 0.999999):
            H.append(_h(f"V-IDENT-{f}", "violacion", "derivada_del_target", f,
                        "la feature es igual al target o una transformación afín exacta de él", excl_como))
            continue
        k = desfase_exacto(df, f, target, kmax)
        if k is not None and k < max(h, 1):
            cuando = "FUTURO" if k < 0 else "pasado insuficiente"
            H.append(_h(f"V-DESFASE-{f}", "violacion", "ventana_cruza_corte", f,
                        f"coincide exactamente con {target} desplazado {k} periodos ({cuando}; se requiere >= {max(h, 1)})", excl_como))

    ya = {x["feature"] for x in H if x["clase"] == "violacion"}
    folds = folds_dev(run)
    u_r2, u_auc, u_corr = float(lk.get("umbral_univariado_r2", 0.95)), float(lk.get("umbral_univariado_auc", 0.97)), float(lk.get("umbral_correlacion", 0.95))
    from scipy.stats import spearmanr
    yv = df[target] if c["problem_type"] != "classification" else df[target].astype("category").cat.codes
    for f in revisadas:
        if f in ya:
            continue
        sc = univariado_oof(df, f, c, folds)
        um = u_auc if c["problem_type"] == "classification" else u_r2
        if sc is not None and sc >= um:
            H.append(_h(f"S-UNIVAR-{f}", "senal", "poder_univariado", f,
                        f"{'AUC' if c['problem_type'] == 'classification' else 'R²'} fuera de muestra usando solo esta feature = {sc:.4f} (umbral {um})",
                        "Investiga su origen: si deriva del target o se conoce después de predecir, exclúyela; si no, justifícala."))
        if pd.api.types.is_numeric_dtype(df[f]):
            m = df[f].notna()
            if m.sum() > 10 and df.loc[m, f].nunique() > 1:
                rho = float(spearmanr(df.loc[m, f], yv[m]).statistic)
                if abs(rho) >= u_corr:
                    H.append(_h(f"S-CORR-{f}", "senal", "correlacion_anomala", f, f"Spearman con el target = {rho:.4f} (umbral {u_corr})",
                                "Investiga su origen; correlación alta no prueba fuga, pero hay que explicarla."))
        pt = perfil_temporal(df, f, target, kmax)
        if pt and pt["max_futuro"] >= 0.9 and pt["max_futuro"] - pt["max_presente_pasado"] >= 0.1:
            H.append(_h(f"S-TEMPORAL-{f}", "senal", "asociacion_temporal", f,
                        f"correlación con el target {pt['k_futuro']} periodos en el FUTURO = {pt['max_futuro']:.3f} vs "
                        f"máx. presente/pasado = {pt['max_presente_pasado']:.3f}",
                        "Verifica cuándo se genera el dato. Si incorpora información posterior al momento de predicción, exclúyelo."))

    # dominancia en importancia por permutación (sobre features no marcadas como violación)
    usables = [f for f in revisadas if f not in ya]
    dom = dominancia_permutacion(df, usables, c, folds[-1]) if usables else None
    if dom and dom["participacion"] >= 0.8:
        s = _h(f"S-IMPORTANCIA-{dom['top']}", "senal", "dominancia_importancia", dom["top"],
               f"concentra {dom['participacion']:.0%} de la importancia por permutación (fold {folds[-1]['id']})",
               "Verifica que no sea una fuga; si es legítima, justifícala.")
        s["severidad"] = "media"
        H.append(s)

    # splits y duplicados
    sp = read_json(run / "splits.json")
    for i, v in enumerate(validar_splits(df, sp, c)):
        H.append(_h(f"V-SPLIT-{i}", "violacion", "split", None, v, "Corrige validacion.* y vuelve a correr contract.py validate."))
    idc = d["id_fila"]
    contenido = [k for k in df.columns if k not in (idc, "_t", "_entidad")]
    firma = pd.util.hash_pandas_object(df[contenido], index=False).astype(str)
    fid = dict(zip(df[idc].astype(str), firma))
    cruces = 0
    for fo in sp["folds"]:
        cruces += len({fid[i] for i in fo["train"]} & {fid[i] for i in fo["val"]})
    if cruces:
        if d["duplicados"] == "accidentales":
            H.append(_h("V-DUP", "violacion", "duplicados", None, f"{cruces} filas idénticas cruzan entrenamiento/validación",
                        "Revisa la deduplicación del snapshot."))
        else:
            s = _h("S-DUP", "senal", "duplicados", None, f"{cruces} filas idénticas (declaradas legítimas) cruzan entrenamiento/validación",
                   "Confirma que son observaciones reales distintas; si no, cambia datos.duplicados a accidentales.")
            s["severidad"] = "media"
            H.append(s)

    # resoluciones
    res_p = run / "leakage_resolutions.yaml"
    res = yaml.safe_load(res_p.read_text(encoding="utf-8")) if res_p.exists() else {}
    res = res or {}
    errores_res = []
    for x in H:
        r = res.get(x["id"])
        if not r:
            continue
        if x["clase"] == "violacion":
            errores_res.append(f"{x['id']}: las violaciones comprobadas no se justifican; hay que corregirlas.")
            x["nota"] = "Se intentó justificar una violación comprobada; la justificación se ignora."
            continue
        j = (r.get("justificacion") or "").strip()
        if r.get("estado") == "justificada" and len(j) >= MIN_JUST:
            x["estado"], x["justificacion"] = "justificada", j
        else:
            errores_res.append(f"{x['id']}: justificación ausente o menor a {MIN_JUST} caracteres.")
    abiertos = [x for x in H if x["estado"] == "abierto"]
    veredicto = "bloqueado" if abiertos else ("aprobado_con_excepciones" if any(x["estado"] == "justificada" for x in H) else "limpio")
    return {
        "veredicto": veredicto, "hallazgos": H, "features_revisadas": revisadas, "features_excluidas": excluidas,
        "errores_resoluciones": errores_res, "permutacion": dom,
        "contract_sha256": sha256_file(run / "contract.yaml"), "dev_sha256": sha256_file(find_table(run / "dev")),
        "resoluciones_sha256": sha256_file(res_p),
        "duplicados_eliminados": len(((read_json(run / "holdout_manifest.json") or {}).get("duplicados") or {}).get("ids_duplicados_eliminados", [])),
        "limites": ("Las señales estadísticas no prueban leakage, y una feature del futuro disfrazada con ruido o "
                    "transformaciones puede no detectarse sin linaje. audit.py repite las pruebas al final."),
    }


def _md(R: dict) -> str:
    L = [f"Veredicto de leakage: {R['veredicto']}", "",
         f"Features revisadas: {len(R['features_revisadas'])}; excluidas por contrato: {R['features_excluidas'] or 'ninguna'}; "
         f"duplicados accidentales eliminados al congelar: {R['duplicados_eliminados']}.", ""]
    if R["hallazgos"]:
        L += ["| id | clase | severidad | estado | evidencia |", "|---|---|---|---|---|"]
        for x in R["hallazgos"]:
            L.append(f"| {x['id']} | {x['clase']} | {x['severidad']} | {x['estado']} | {x['evidencia']} |")
    else:
        L.append("Sin hallazgos.")
    for e in R["errores_resoluciones"]:
        L.append(f"\n- AVISO: {e}")
    L.append(f"\nLímite: {R['limites']}")
    return "\n".join(L)


def criterios(run):
    """Criterios de cierre de la fase EDA (los usa gate.py advance eda)."""
    from profile_data import INTERP, MARCA_LEAK
    run = Path(run)
    P, rels = [], ["eda/profile.json", "eda_report.md", "leakage_report.json"]
    prof = read_json(run / "eda" / "profile.json")
    dev_h = sha256_file(find_table(run / "dev"))
    if prof is None:
        P.append(("falta eda/profile.json.", f"Corre `python scripts/profile_data.py --run {run}`."))
    elif prof.get("dev_sha256") != dev_h:
        P.append(("eda/profile.json corresponde a otros datos de desarrollo.", f"Vuelve a correr profile_data.py."))
    R = read_json(run / "leakage_report.json")
    if R is None:
        P.append(("falta leakage_report.json.", f"Corre `python scripts/leakage_check.py --run {run}`."))
    else:
        res_p = run / "leakage_resolutions.yaml"
        if (R["contract_sha256"] != sha256_file(run / "contract.yaml") or R["dev_sha256"] != dev_h
                or R["resoluciones_sha256"] != sha256_file(res_p)):
            P.append(("leakage_report.json está desactualizado (cambió el contrato, los datos o las resoluciones).",
                      "Vuelve a correr leakage_check.py."))
        if R["veredicto"] == "bloqueado":
            ab = [x["id"] for x in R["hallazgos"] if x["estado"] == "abierto"]
            P.append((f"leakage bloqueado; hallazgos abiertos: {ab}.",
                      "Excluye las features (contract.yaml + decisions.md) o justifica las SEÑALES en leakage_resolutions.yaml; "
                      "las violaciones solo se corrigen."))
        if res_p.exists():
            rels.append("leakage_resolutions.yaml")
    rep = run / "eda_report.md"
    if not rep.exists():
        P.append(("falta eda_report.md.", "Corre profile_data.py."))
    else:
        txt = rep.read_text(encoding="utf-8")
        if R is not None and (f"Veredicto de leakage: {R['veredicto']}" not in txt or MARCA_LEAK[0] not in txt):
            P.append(("eda_report.md no declara el veredicto vigente de leakage.", "Vuelve a correr leakage_check.py (actualiza el bloque)."))
        if INTERP not in txt:
            P.append(("eda_report.md no tiene sección de Interpretación.", f"Agrega `{INTERP}` con tu interpretación."))
        else:
            interp = txt.split(INTERP, 1)[1]
            import re
            if "PENDIENTE" in interp or len(interp.strip()) < 200:
                P.append(("la Interpretación del EDA está pendiente o es demasiado corta (< 200 caracteres).",
                          "Escribe qué significan los hechos para el modelado, riesgos y preguntas abiertas."))
            if not re.search(r"(eda/|profile\.json|leakage_report|figuras/)", interp):
                P.append(("la Interpretación no cita ningún artefacto verificable.",
                          "Referencia eda/profile.json, eda/figuras/... o leakage_report.json al sustentar cada afirmación."))
    return P, rels


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    a = ap.parse_args()
    from gate import requerir
    from profile_data import MARCA_LEAK, reemplazar_bloque
    run = Path(a.run)
    requerir(run, "eda")
    if not (run / "eda_report.md").exists():
        raise HarnessError("no existe eda_report.md.", f"Corre primero `python scripts/profile_data.py --run {run}`.")
    R = revisar(run)
    atomic_write_json(run / "leakage_report.json", R)
    md = _md(R)
    atomic_write_text(run / "leakage_report.md", "# Revisión de leakage\n\n" + md + "\n")
    rep = run / "eda_report.md"
    atomic_write_text(rep, reemplazar_bloque(rep.read_text(encoding="utf-8"), MARCA_LEAK, md))
    print(md)
    if R["veredicto"] == "bloqueado":
        print("\nBLOQUEADO: hay hallazgos de leakage abiertos. Cómo resolverlo: ver columna 'accion' en leakage_report.json.",
              flush=True)
        return 1
    return 0


if __name__ == "__main__":
    cli(main)
