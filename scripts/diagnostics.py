"""
Fase 4 — Diagnóstico del modelo elegido (sobre sus predicciones fuera de muestra).

Uso:
    python scripts/diagnostics.py --run runs/<p> [--max-folds-permutacion 3]
    python scripts/gate.py advance diagnostico --run runs/<p>

Cada diagnóstico produce un resultado estructurado:
    accionable     hay algo concreto que hacer (con evidencia)
    sin_hallazgos  se evaluó y no hay nada que hacer (con evidencia)
    no_evaluable   no aplica o no se pudo evaluar (con justificación)

Diagnósticos:
  todos           error por segmento (diagnostico.segmentos del contrato), peores errores,
                  importancia por permutación (re-ajustando por fold, con la métrica principal),
                  ablaciones de grupos de features (diagnostico.grupos_features).
  regresión       sesgo de los residuos (IC por folds) y heterocedasticidad (Breusch–Pagan).
  forecasting     lo anterior + error por horizonte, por entidad y por periodo (fold), y
                  autocorrelación de residuos (Ljung–Box por entidad).
  clasificación   calibración (ECE, diagrama de confiabilidad), errores por clase y sensibilidad al umbral.

Escribe diagnostics/diagnostics.json, diagnostics/diagnostics.md y figuras. Si ningún resultado es
accionable, el gate exige diagnostics/declaracion.md explicando por qué no hay nada que hacer.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import HarnessError, ahora, atomic_write_json, atomic_write_text, cli, read_json, read_table, sha256_file
from metrics import METRICAS, calcular

RESULTADOS = ("accionable", "sin_hallazgos", "no_evaluable")


def _r(id_, nombre, resultado, evidencia, justificacion=None, figura=None, datos=None):
    return {"id": id_, "nombre": nombre, "resultado": resultado, "evidencia": evidencia,
            "justificacion": justificacion, "figura": figura, "datos": datos}


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _metrica_grupo(g: pd.DataFrame, pm: str, clases) -> float:
    P = g[[f"p__{k}" for k in clases]].values if clases and f"p__{clases[0]}" in g else None
    return calcular(pm, g["y_true"].values, g["y_pred"].values, P, clases)


def por_segmento(preds, dev, c, pm, clases, margen) -> list[dict]:
    out = []
    idc = c["datos"]["id_fila"]
    segs = list((c.get("diagnostico") or {}).get("segmentos") or [])
    if not segs:
        return [_r("D-SEG", "error por segmento", "no_evaluable", "no hay segmentos declarados",
                   "El contrato no declara diagnostico.segmentos; decláralos si hay subgrupos de negocio relevantes.")]
    met = METRICAS[pm]
    glob = _metrica_grupo(preds, pm, clases)
    p = preds.merge(dev[[idc, *[s for s in segs if s in dev]]].rename(columns={idc: "id"}).astype({"id": str}), on="id", how="left")
    for s in segs:
        if s not in p:
            out.append(_r(f"D-SEG-{s}", f"error por segmento {s}", "no_evaluable", f"la columna {s} no existe en desarrollo",
                          "Corrige diagnostico.segmentos."))
            continue
        tab = {str(k): (_metrica_grupo(g, pm, clases), len(g)) for k, g in p.groupby(s) if len(g) >= 20}
        if not tab:
            out.append(_r(f"D-SEG-{s}", f"error por segmento {s}", "no_evaluable", "ningún segmento con >= 20 filas", "Muestra insuficiente."))
            continue
        signo = 1 if met.mayor_mejor else -1
        peores = {k: v for k, (v, n) in tab.items() if signo * (glob - v) > max(abs(glob) * 0.25, margen)}
        datos = {k: {"valor": v, "n": n} for k, (v, n) in tab.items()}
        if peores:
            out.append(_r(f"D-SEG-{s}", f"error por segmento {s}", "accionable",
                          f"{pm} global {glob:.4g}; segmentos con desempeño >25% peor (y más allá del margen): "
                          + ", ".join(f"{k}={v:.4g}" for k, v in peores.items()), datos=datos))
        else:
            out.append(_r(f"D-SEG-{s}", f"error por segmento {s}", "sin_hallazgos",
                          f"{pm} global {glob:.4g}; rango por segmento [{min(v for v, _ in tab.values()):.4g}, {max(v for v, _ in tab.values()):.4g}]",
                          datos=datos))
    return out


def peores_errores(preds, c, clases) -> dict:
    p = preds[preds["semilla"] == preds["semilla"].min()].copy()
    if clases:
        idx = {k: i for i, k in enumerate(clases)}
        P = p[[f"p__{k}" for k in clases]].values if f"p__{clases[0]}" in p else None
        if P is None:
            return _r("D-PEORES", "peores errores", "no_evaluable", "sin probabilidades", "El modelo no produce probabilidades.")
        p["error"] = -np.log(np.clip(P[np.arange(len(p)), [idx[v] for v in p["y_true"]]], 1e-15, 1))
    else:
        p["error"] = (p["y_true"] - p["y_pred"]).abs()
    top = p.nlargest(20, "error")
    ev = f"top 20 errores: media {top['error'].mean():.4g} vs mediana global {p['error'].median():.4g}"
    datos = top[["id", "fold", "y_true", "y_pred", "error"] + (["entidad"] if "entidad" in top else [])].head(10).to_dict("records")
    if "entidad" in top:
        frac = top["entidad"].value_counts(normalize=True)
        if frac.iloc[0] >= 0.5:
            return _r("D-PEORES", "peores errores", "accionable",
                      ev + f"; {frac.iloc[0]:.0%} de los 20 peores son de la entidad {frac.index[0]}: revisar datos/comportamiento de esa entidad.",
                      datos=datos)
    q = p["error"].quantile(0.99) / max(p["error"].median(), 1e-12)
    if q > 10:
        return _r("D-PEORES", "peores errores", "accionable",
                  ev + f"; el percentil 99 del error es {q:.1f}x la mediana: hay casos extremos que revisar (outliers o datos erróneos).",
                  datos=datos)
    return _r("D-PEORES", "peores errores", "sin_hallazgos", ev + "; sin concentración por entidad ni colas extremas.", datos=datos)


def residuos(preds, c, margen, figdir) -> list[dict]:
    from scipy import stats
    out = []
    p = preds.groupby(["fold", "id"], as_index=False)[["y_true", "y_pred"]].mean()
    p["res"] = p["y_true"] - p["y_pred"]
    por_fold = p.groupby("fold")["res"].mean()
    m, se = por_fold.mean(), por_fold.std(ddof=1) / np.sqrt(len(por_fold)) if len(por_fold) > 1 else np.nan
    tq = stats.t.ppf(0.975, max(len(por_fold) - 1, 1))
    ic = [m - tq * se, m + tq * se]
    if np.isfinite(se) and (ic[0] > 0 or ic[1] < 0) and abs(m) > 0.1 * margen:
        out.append(_r("D-SESGO", "sesgo de residuos", "accionable",
                      f"residuo medio {m:.4g} (IC95 por folds [{ic[0]:.4g}, {ic[1]:.4g}]): el modelo {'sub' if m > 0 else 'sobre'}estima sistemáticamente.",
                      datos={"media": m, "ic95": ic}))
    else:
        out.append(_r("D-SESGO", "sesgo de residuos", "sin_hallazgos", f"residuo medio {m:.4g} (IC95 por folds [{ic[0]:.4g}, {ic[1]:.4g}]).",
                      datos={"media": m, "ic95": ic}))
    import statsmodels.api as sm
    from statsmodels.stats.diagnostic import het_breuschpagan
    X = sm.add_constant(p["y_pred"].values)
    lm, pv, _, _ = het_breuschpagan(p["res"].values, X)
    rho = stats.spearmanr(p["y_pred"], p["res"].abs()).statistic
    res = "accionable" if pv < 0.01 and abs(rho) > 0.1 else "sin_hallazgos"
    out.append(_r("D-HETERO", "heterocedasticidad", res,
                  f"Breusch–Pagan p={pv:.3g}; Spearman(|residuo|, predicción)={rho:.3f}"
                  + ("; la varianza del error cambia con el nivel: considerar transformar el target o reportar intervalos condicionados." if res == "accionable" else ""),
                  figura="residuos.png"))
    plt = _plt()
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    ax[0].scatter(p["y_pred"], p["res"], s=4, alpha=0.4)
    ax[0].axhline(0, color="k", lw=0.8)
    ax[0].set_xlabel("predicción"); ax[0].set_ylabel("residuo"); ax[0].set_title("Residuos vs predicción")
    ax[1].hist(p["res"], bins=40); ax[1].set_title("Distribución de residuos")
    plt.tight_layout(); plt.savefig(figdir / "residuos.png", dpi=90); plt.close()
    return out


def forecasting_diag(preds, c, pm, margen, figdir) -> list[dict]:
    from statsmodels.stats.diagnostic import acorr_ljungbox
    out = []
    p = preds[preds["semilla"] == preds["semilla"].min()]
    met = METRICAS[pm]
    signo = 1 if met.mayor_mejor else -1
    hz = {int(h): _metrica_grupo(g, pm, None) for h, g in p.groupby("horizonte")}
    hs = sorted(hz)
    deg = signo * (hz[hs[0]] - hz[hs[-1]])
    res = "accionable" if deg > max(margen, 0.3 * abs(hz[hs[0]])) else "sin_hallazgos"
    out.append(_r("D-HORIZONTE", "error por horizonte", res,
                  f"{pm} en h={hs[0]}: {hz[hs[0]]:.4g}; en h={hs[-1]}: {hz[hs[-1]]:.4g}"
                  + ("; el error crece con el horizonte: considerar modelos por horizonte o reducir el horizonte operativo." if res == "accionable" else ""),
                  figura="error_horizonte.png", datos=hz))
    plt = _plt()
    plt.figure(figsize=(5, 3)); plt.plot(hs, [hz[h] for h in hs], marker="o"); plt.xlabel("horizonte"); plt.ylabel(pm)
    plt.title("Error por horizonte"); plt.tight_layout(); plt.savefig(figdir / "error_horizonte.png", dpi=90); plt.close()
    pf = {f: _metrica_grupo(g, pm, None) for f, g in p.groupby("fold")}
    vals = np.array(list(pf.values()))
    tend = np.polyfit(np.arange(len(vals)), vals, 1)[0] if len(vals) > 2 else 0
    res = "accionable" if signo * -tend * len(vals) > max(margen, 0.2 * abs(vals.mean())) else "sin_hallazgos"
    out.append(_r("D-PERIODO", "error por periodo (ventana)", res,
                  f"{pm} por ventana: " + ", ".join(f"{k}={v:.4g}" for k, v in pf.items())
                  + ("; el error empeora en ventanas recientes: posible drift, reentrenar con más frecuencia." if res == "accionable" else ""),
                  datos=pf))
    if "entidad" in p:
        pv = []
        for e, g in p.sort_values("t").groupby("entidad"):
            r = (g["y_true"] - g["y_pred"]).values
            if len(r) > 10:
                lags = [1] + ([int(c["prediccion"]["estacionalidad"])] if c["prediccion"].get("estacionalidad") and len(r) > 2 * int(c["prediccion"]["estacionalidad"]) else [])
                pv.append(float(acorr_ljungbox(r, lags=lags)["lb_pvalue"].min()))
        if pv:
            frac = float(np.mean(np.array(pv) < 0.05))
            res = "accionable" if frac > 0.5 else "sin_hallazgos"
            out.append(_r("D-AUTOCORR", "autocorrelación de residuos", res,
                          f"Ljung–Box p<0.05 en {frac:.0%} de las entidades (residuos concatenados de las ventanas)"
                          + ("; queda estructura temporal sin modelar (rezagos/estacionalidad)." if res == "accionable" else ""),
                          datos={"pvalores": pv}))
    return out


def clasificacion_diag(preds, c, clases, figdir) -> list[dict]:
    out = []
    p = preds[preds["semilla"] == preds["semilla"].min()]
    if f"p__{clases[0]}" not in p:
        return [_r("D-CALIB", "calibración", "no_evaluable", "sin probabilidades", "El modelo no produce probabilidades.")]
    plt = _plt()
    if len(clases) == 2:
        y = (p["y_true"] == clases[1]).astype(int).values
        q = p[f"p__{clases[1]}"].values
        bins = np.clip((q * 10).astype(int), 0, 9)
        ece, curva = 0.0, []
        for b in range(10):
            mk = bins == b
            if mk.any():
                ece += mk.mean() * abs(y[mk].mean() - q[mk].mean())
                curva.append((float(q[mk].mean()), float(y[mk].mean()), int(mk.sum())))
        res = "accionable" if ece > 0.05 else "sin_hallazgos"
        out.append(_r("D-CALIB", "calibración", res, f"ECE (10 bins) = {ece:.4f}"
                      + ("; calibrar dentro del CV (p. ej. CalibratedClassifierCV en el pipeline)." if res == "accionable" else ""),
                      figura="calibracion.png", datos={"curva": curva}))
        plt.figure(figsize=(4, 4)); plt.plot([0, 1], [0, 1], "k--", lw=0.8)
        plt.plot([a for a, _, _ in curva], [b for _, b, _ in curva], marker="o"); plt.xlabel("probabilidad predicha")
        plt.ylabel("frecuencia observada"); plt.title("Confiabilidad"); plt.tight_layout(); plt.savefig(figdir / "calibracion.png", dpi=90); plt.close()
        # sensibilidad al umbral (informativa: ajustar el umbral fuera del CV sería fuga)
        f1s = {round(u, 2): calcular("f1", p["y_true"].values, np.where(q >= u, clases[1], clases[0]), None, clases) for u in np.linspace(0.05, 0.95, 19)}
        u_act = float((c["metrica"].get("umbral") or {}).get("valor", 0.5))
        u_best = max(f1s, key=f1s.get)
        f_act = calcular("f1", p["y_true"].values, p["y_pred"].values, None, clases)
        res = "accionable" if f1s[u_best] - f_act > 0.05 else "sin_hallazgos"
        out.append(_r("D-UMBRAL", "sensibilidad al umbral", res,
                      f"F1 con el umbral usado ({u_act}): {f_act:.3f}; máximo en OOF {f1s[u_best]:.3f} en {u_best}"
                      + ("; si el F1 importa, usar metrica.umbral.politica = ajustada_interna (se ajusta dentro de cada fold). "
                         "Elegir el umbral mirando estas predicciones sería sobreajuste." if res == "accionable" else ""),
                      datos=f1s))
    acc = calcular("accuracy", p["y_true"].values, p["y_pred"].values, None, clases)
    bacc = calcular("balanced_accuracy", p["y_true"].values, p["y_pred"].values, None, clases)
    rec = {str(k): float(((p["y_pred"] == k) & (p["y_true"] == k)).sum() / max((p["y_true"] == k).sum(), 1)) for k in clases}
    res = "accionable" if acc - bacc > 0.1 else "sin_hallazgos"
    out.append(_r("D-CLASES", "errores por clase", res, f"accuracy {acc:.3f} vs balanced accuracy {bacc:.3f}; recall por clase {rec}"
                  + ("; el modelo descuida la clase minoritaria con el umbral actual." if res == "accionable" else ""), datos=rec))
    return out


def permutacion(run, K, spec, rec, pm, max_folds, figdir) -> dict:
    from experiment_log import ajustar_predecir
    c, dev = K["c"], K["dev"]
    feats = spec["features"]
    if not feats or rec["familia"] == "ingenuo":
        return _r("D-PERM", "importancia por permutación", "no_evaluable", "el modelo elegido no usa features",
                  "Modelo ingenuo: no hay features que permutar.")
    met = METRICAS[pm]
    signo = 1 if met.mayor_mejor else -1
    folds = K["sp"]["folds"]
    folds = folds[-max_folds:] if K["sp"]["esquema"] == "temporal" else folds[:max_folds]
    rng = np.random.RandomState(0)
    imp = {f: [] for f in feats}
    s0 = rec["semillas"][0]
    for fo in folds:
        tr, va = dev.iloc[K["pos"][fo["train"]].values], dev.iloc[K["pos"][fo["val"]].values]
        r = ajustar_predecir(spec, c, tr, va, s0, K["clases"], K["umbral"])
        est = r["estimador"]
        base = calcular(pm, va[c["datos"]["target"]].values, r["yhat"], r["P"], K["clases"])
        cols = list(getattr(est, "feature_names_in_", feats))
        for f in feats:
            for _ in range(3):
                vp = va.copy()
                vp[f] = rng.permutation(vp[f].values)
                X = vp[[k for k in cols if k in vp]]
                if K["clases"]:
                    from experiment_log import _proba
                    P = _proba(est, X, K["clases"])
                    yh = np.where(P[:, 1] >= (r["umbral"] or 0.5), K["clases"][1], K["clases"][0]) if len(K["clases"]) == 2 else np.array(K["clases"])[P.argmax(1)]
                else:
                    P, yh = None, est.predict(X)
                v = calcular(pm, va[c["datos"]["target"]].values, yh, P, K["clases"])
                imp[f].append(signo * (base - v))
    tab = {f: float(np.mean(v)) for f, v in imp.items()}
    tab = dict(sorted(tab.items(), key=lambda kv: -kv[1]))
    pos = {k: v for k, v in tab.items() if v > 0}
    tot = sum(pos.values())
    inutiles = [k for k, v in tab.items() if v <= 0]
    plt = _plt()
    plt.figure(figsize=(5, 0.3 * len(tab) + 1)); pd.Series(tab).sort_values().plot.barh(); plt.title(f"Permutación (Δ {pm})")
    plt.tight_layout(); plt.savefig(figdir / "permutacion.png", dpi=90); plt.close()
    msgs = []
    if tot > 0 and max(pos.values()) / tot > 0.6:
        top = max(pos, key=pos.get)
        msgs.append(f"{top} concentra {pos[top] / tot:.0%} de la importancia: verificar que no sea fuga y evaluar dependencia operativa")
    if inutiles:
        msgs.append(f"features sin aporte al permutarlas: {inutiles} (candidatas a eliminar; confirmar con ablación)")
    return _r("D-PERM", "importancia por permutación", "accionable" if msgs else "sin_hallazgos",
              ("; ".join(msgs) if msgs else "importancia repartida; todas las features aportan") + f" (folds {[f['id'] for f in folds]}, 3 repeticiones)",
              figura="permutacion.png", datos=tab)


def ablaciones(run, K, spec, rec, pm, margen) -> list[dict]:
    from experiment_log import correr_cv, resumir
    grupos = (K["c"].get("diagnostico") or {}).get("grupos_features") or {}
    if not grupos:
        return [_r("D-ABL", "ablaciones", "no_evaluable", "no hay grupos de features declarados",
                   "Declara diagnostico.grupos_features en el contrato para medir el aporte de cada grupo.")]
    if not spec["features"] or rec["familia"] == "ingenuo":
        return [_r("D-ABL", "ablaciones", "no_evaluable", "el modelo elegido no usa features", "Modelo ingenuo.")]
    met = METRICAS[pm]
    signo = 1 if met.mayor_mejor else -1
    s0 = rec["semillas"][:1]
    base = pd.DataFrame(rec["metricas_por_fold"])
    base_val = float(base[base["semilla"] == s0[0]].groupby("fold")[pm].mean().mean())
    out = []
    for nombre, cols in grupos.items():
        quitar = [f for f in cols if f in spec["features"]]
        if not quitar:
            out.append(_r(f"D-ABL-{nombre}", f"ablación {nombre}", "no_evaluable", "el modelo no usa ninguna feature del grupo", "No aplica."))
            continue
        sp2 = dict(spec, features=[f for f in spec["features"] if f not in quitar])
        if not sp2["features"]:
            out.append(_r(f"D-ABL-{nombre}", f"ablación {nombre}", "no_evaluable", "quitar el grupo deja el modelo sin features", "No aplica."))
            continue
        try:
            R = correr_cv(K, sp2, s0)
        except Exception as e:
            out.append(_r(f"D-ABL-{nombre}", f"ablación {nombre}", "no_evaluable", f"el pipeline no admite quitar features: {e}",
                          "El build_pipeline depende de nombres fijos de columnas; usa selectores por tipo (modelos.py) para permitir ablaciones."))
            continue
        v = float(pd.DataFrame(R["filas"]).groupby("fold")[pm].mean().mean())
        deg = signo * (base_val - v)
        if deg <= margen:
            out.append(_r(f"D-ABL-{nombre}", f"ablación {nombre}", "accionable",
                          f"sin {quitar}: {pm} {v:.4g} vs {base_val:.4g} (degradación {deg:.4g} <= margen {margen}): el grupo no aporta más que el margen; "
                          "candidato a simplificar (requiere un experimento nuevo registrado).", datos={"sin_grupo": v, "base": base_val}))
        else:
            out.append(_r(f"D-ABL-{nombre}", f"ablación {nombre}", "sin_hallazgos",
                          f"sin {quitar}: {pm} {v:.4g} vs {base_val:.4g} (degradación {deg:.4g} > margen): el grupo aporta.",
                          datos={"sin_grupo": v, "base": base_val}))
    return out


def diagnosticar(run, max_folds_perm=3) -> dict:
    from experiment_log import bitacora, contexto_cv, spec_desde_registro
    run = Path(run)
    cmpj = read_json(run / "comparison.json")
    sel = cmpj["congelado"]
    rec = [r for r in bitacora(run) if r.get("run_id") == sel["run_id"] and r.get("tipo") in ("baseline", "experimento")][-1]
    K = contexto_cv(run)
    c = K["c"]
    pm = c["metrica"]["principal"]
    margen = float(c["comparacion"]["margen_practico"])
    figdir = run / "diagnostics" / "figuras"
    figdir.mkdir(parents=True, exist_ok=True)
    preds = read_table(run / rec["preds_ref"])
    spec = spec_desde_registro(run, rec)
    R = []
    R += por_segmento(preds, K["dev"], c, pm, K["clases"], margen)
    R.append(peores_errores(preds, c, K["clases"]))
    if c["problem_type"] == "classification":
        R += clasificacion_diag(preds, c, K["clases"], figdir)
    else:
        R += residuos(preds, c, margen, figdir)
        if c["problem_type"] == "forecasting":
            R += forecasting_diag(preds, c, pm, margen, figdir)
    R.append(permutacion(run, K, spec, rec, pm, max_folds_perm, figdir))
    R += ablaciones(run, K, spec, rec, pm, margen)
    return {"fecha": ahora(), "seleccionado": sel["run_id"], "comparison_sha256": sha256_file(run / "comparison.json"),
            "metrica": pm, "resultados": R,
            "resumen": {k: sum(1 for x in R if x["resultado"] == k) for k in RESULTADOS}}


def _md(D: dict) -> str:
    L = [f"# Diagnóstico de {D['seleccionado']}", "", f"Resumen: {D['resumen']}", "",
         "| id | diagnóstico | resultado | evidencia |", "|---|---|---|---|"]
    for x in D["resultados"]:
        ev = x["evidencia"] + (f" — Justificación: {x['justificacion']}" if x.get("justificacion") else "")
        L.append(f"| {x['id']} | {x['nombre']} | **{x['resultado']}** | {ev} |")
    for x in D["resultados"]:
        if x.get("figura"):
            L.append(f"\n![{x['nombre']}](figuras/{x['figura']})")
    return "\n".join(L)


REQUERIDOS = {
    "regression": ["D-SEG", "D-PEORES", "D-SESGO", "D-HETERO", "D-PERM", "D-ABL"],
    "forecasting": ["D-SEG", "D-PEORES", "D-SESGO", "D-HETERO", "D-HORIZONTE", "D-PERIODO", "D-PERM", "D-ABL"],
    "classification": ["D-SEG", "D-PEORES", "D-CALIB", "D-CLASES", "D-PERM", "D-ABL"],
}


def criterios(run):
    from contract import cargar_contrato
    run = Path(run)
    D = read_json(run / "diagnostics" / "diagnostics.json")
    if D is None:
        return [("falta diagnostics/diagnostics.json.", f"Corre `python scripts/diagnostics.py --run {run}`.")], []
    P = []
    rels = ["diagnostics/diagnostics.json"]
    if D["comparison_sha256"] != sha256_file(run / "comparison.json"):
        P.append(("el diagnóstico corresponde a otra comparación/selección.", "Vuelve a correr diagnostics.py."))
    pt = cargar_contrato(run)["problem_type"]
    ids = [x["id"] for x in D["resultados"]]
    falt = [k for k in REQUERIDOS[pt] if not any(i == k or i.startswith(k + "-") for i in ids)]
    if falt:
        P.append((f"faltan diagnósticos requeridos: {falt}.", "Vuelve a correr diagnostics.py."))
    for x in D["resultados"]:
        if x["resultado"] not in RESULTADOS or not x.get("evidencia"):
            P.append((f"{x['id']}: resultado inválido o sin evidencia.", "Vuelve a correr diagnostics.py."))
        if x["resultado"] == "no_evaluable" and not x.get("justificacion"):
            P.append((f"{x['id']}: no_evaluable sin justificación.", "Vuelve a correr diagnostics.py."))
    if not any(x["resultado"] == "accionable" for x in D["resultados"]):
        dec = run / "diagnostics" / "declaracion.md"
        if not dec.exists() or len(dec.read_text(encoding="utf-8").strip()) < 150:
            P.append(("ningún diagnóstico es accionable y no hay declaración explícita.",
                      "Escribe diagnostics/declaracion.md (>= 150 caracteres) explicando por qué no hay nada que hacer, citando diagnostics.json."))
        else:
            rels.append("diagnostics/declaracion.md")
    return P, rels


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--max-folds-permutacion", type=int, default=3)
    a = ap.parse_args()
    from gate import requerir
    requerir(a.run, "diagnostico")
    D = diagnosticar(a.run, a.max_folds_permutacion)
    atomic_write_json(Path(a.run) / "diagnostics" / "diagnostics.json", D)
    md = _md(D)
    atomic_write_text(Path(a.run) / "diagnostics" / "diagnostics.md", md + "\n")
    print(md)
    return 0


if __name__ == "__main__":
    cli(main)
