"""
Fase 3 — Comparación de candidatos y elección determinista del modelo.

Uso:
    python scripts/compare_models.py --run runs/<p>
    python scripts/compare_models.py --run runs/<p> --override E005 --justificacion "..."
    python scripts/gate.py advance experimentos --run runs/<p>

Todos los candidatos (baselines + experimentos vigentes) usan los mismos folds. La métrica
principal, su agregación y el margen práctico vienen del contrato (fijados antes de experimentar).

Inferencia según el diseño (`validacion.esquema`):
  kfold / estratificado  t pareada corregida de Nadeau–Bengio sobre las diferencias por fold
                         (promediadas sobre semillas). Supone folds intercambiables; la corrección
                         (1/J + n_val/n_train) compensa el solapamiento de los entrenamientos.
  grupos                 bootstrap pareado por ENTIDAD dentro de cada fold (la entidad es la
                         unidad de dependencia).
  temporal               bootstrap pareado de BLOQUES de periodos dentro de cada ventana; cada
                         bloque lleva a todas las entidades de esos periodos (preserva la
                         dependencia entre entidades).
  Con menos de `comparacion.min_unidades` unidades (o J < 5 folds en Nadeau–Bengio) el resultado
  es `inconcluso`: no se fabrica precisión.
  Las semillas NO son observaciones: dentro de cada réplica se promedia la métrica sobre semillas.
  Un bootstrap ordinario de scores por fold no se usa.

Política de elección (determinista):
  1. Excluir candidatos inválidos (features ya no permitidas, fallidos) o que incumplen restricciones.
  2. Mejor = mejor valor estimado de la métrica principal.
  3. Para cada candidato, degradación vs el mejor (en unidades de la métrica) y pruebas
     unilaterales de no inferioridad (H0: deg >= margen) e inferioridad (H0: deg <= margen),
     con corrección de Holm sobre la familia explícita "cada candidato vs el mejor".
  4. Elegido = el más simple (comparacion.orden_simplicidad, luego menos features) entre los
     no inferiores.
  5. Si un candidato más simple queda inconcluso, la elección es `provisional` y se aplica
     `comparacion.desempate`. Sin inferencia válida, la elección es `operativa` (por estimación
     puntual y restricciones) y así se etiqueta.
"No significativo" no equivale a "equivalente". Holm controla la familia declarada; NO corrige el
sesgo de búsqueda adaptativa (muchos experimentos sobre los mismos folds): para afirmaciones
confirmatorias está el holdout, evaluado una sola vez.

`--override` exige una entrada en decisions.md que mencione el id, y no permite elegir candidatos
inválidos ni con features que violan el contrato.
"""
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from common import HarnessError, ahora, atomic_write_json, atomic_write_text, cli, read_json, read_table, sha256_file
from metrics import METRICAS, preparar

SEMILLA_BOOT = 20261022


# ------------------------------------------------------------------ estadística

def holm(pvals: list[float]) -> list[float]:
    """Ajuste de Holm–Bonferroni (devuelve p ajustados en el orden original)."""
    m = len(pvals)
    if m == 0:
        return []
    orden = np.argsort(pvals)
    adj = np.empty(m)
    acum = 0.0
    for rank, i in enumerate(orden):
        acum = max(acum, min(1.0, (m - rank) * pvals[i]))
        adj[i] = acum
    return adj.tolist()


def nadeau_bengio(d: np.ndarray, ratio: float, margen: float, alfa: float) -> dict:
    """t corregida para diferencias por fold `d` (deg candidato - mejor). ratio = n_val / n_train."""
    J = len(d)
    media = float(np.mean(d))
    var = float(np.var(d, ddof=1)) if J > 1 else 0.0
    se = math.sqrt((1.0 / J + ratio) * var)
    if se == 0:
        p_ni = 0.0 if media < margen else 1.0
        p_inf = 0.0 if media > margen else 1.0
        return {"deg": media, "ic": [media, media], "p_noinf": p_ni, "p_inf": p_inf, "se": 0.0}
    tq = stats.t.ppf(1 - alfa / 2, J - 1)
    z = (media - margen) / se
    return {"deg": media, "ic": [float(media - tq * se), float(media + tq * se)], "p_noinf": float(stats.t.cdf(z, J - 1)),
            "p_inf": float(stats.t.sf(z, J - 1)), "se": se}


def _unidades(preds: pd.DataFrame, esquema: str, L: int) -> pd.Series:
    if esquema == "temporal":
        tmin = preds.groupby("fold")["t"].transform("min")
        return preds["fold"] + "|" + ((preds["t"] - tmin) // L).astype(int).astype(str)
    if esquema == "grupos":
        return preds["fold"] + "|" + preds["entidad"].astype(str)
    return preds["fold"] + "|" + preds["id"].astype(str)   # filas iid (solo se usa en el holdout)


class Bootstrap:
    """Bootstrap pareado: las mismas réplicas (conteos por unidad dentro de cada fold) para todos los modelos."""

    def __init__(self, ref: pd.DataFrame, esquema: str, L: int, B: int, rep_por_fold: dict):
        u = _unidades(ref, esquema, L)
        self.esquema, self.L, self.B, self.rep = esquema, L, B, rep_por_fold
        rng = np.random.RandomState(SEMILLA_BOOT)
        self.units, self.counts = {}, {}
        for f in sorted(ref["fold"].unique()):
            lista = sorted(u[ref["fold"] == f].unique())
            self.units[f] = {k: i for i, k in enumerate(lista)}
            self.counts[f] = rng.multinomial(len(lista), np.ones(len(lista)) / len(lista), size=B).astype(float)
        self.n_unidades = sum(len(v) for v in self.units.values())

    def distribucion(self, preds: pd.DataFrame, metrica: str, clases, agregacion: str) -> np.ndarray:
        met = METRICAS[metrica]
        preds = preds.assign(_u=_unidades(preds, self.esquema, self.L))
        pcols = [f"p__{k}" for k in clases] if clases else []
        por_semilla = []
        for _, ps in preds.groupby("semilla"):
            fold_vals = {}   # fold -> (B,) valor  o  (B,k) sumas
            for f, pf in ps.groupby("fold"):
                idx = pf["_u"].map(self.units[f]).values
                P = pf[pcols].values if pcols and pcols[0] in pf else None
                C = self.counts[f]
                if met.componentes is not None:
                    y, yh, PP, K = preparar(metrica, pf["y_true"].values, pf["y_pred"].values, P, clases)
                    comps = met.componentes(y, yh, PP, K)
                    U = np.zeros((len(self.units[f]), comps.shape[1]))
                    np.add.at(U, idx, comps)
                    fold_vals[f] = C @ U
                else:
                    y, yh, PP, K = preparar(metrica, pf["y_true"].values, pf["y_pred"].values, P, clases)
                    filas_u = [np.where(idx == i)[0] for i in range(len(self.units[f]))]
                    vals = np.empty(self.B)
                    for b in range(self.B):
                        sel = np.concatenate([np.repeat(filas_u[i], int(n)) for i, n in enumerate(C[b]) if n])
                        vals[b] = met.directa(y[sel], yh[sel], None if PP is None else PP[sel], K)
                    fold_vals[f] = vals
            reps: dict[int, list] = {}
            for f, v in fold_vals.items():
                reps.setdefault(self.rep[f], []).append(v)
            por_rep = []
            for r, vs in reps.items():
                if met.componentes is not None and agregacion == "pooled":
                    por_rep.append(met.de_sumas(np.sum(vs, axis=0)))
                elif met.componentes is not None:
                    por_rep.append(np.nanmean([met.de_sumas(v) for v in vs], axis=0))
                else:
                    por_rep.append(np.nanmean(vs, axis=0))
            por_semilla.append(np.nanmean(por_rep, axis=0))
        return np.nanmean(por_semilla, axis=0)


# ------------------------------------------------------------------ comparación

def _rango(familia: str, orden: list[str]) -> int:
    return orden.index(familia) if familia in orden else len(orden)


def comparar(run, override: str | None = None, justificacion: str | None = None) -> dict:
    from contract import cargar_contrato, cargar_dev, contexto_eval, estado_feature, features_utilizables
    from experiment_log import registros_vigentes
    run = Path(run)
    c = cargar_contrato(run)
    ctx = contexto_eval(run)
    m, cmp_, rs = c["metrica"], c.get("comparacion") or {}, c.get("restricciones") or {}
    pm = m["principal"]
    met = METRICAS[pm]
    margen = float(cmp_["margen_practico"])
    alfa = float(cmp_.get("alfa", 0.05))
    orden = list(cmp_.get("orden_simplicidad") or ["ingenuo", "lineal", "arbol", "gbm", "ensamble", "otro"])
    esquema = c["validacion"]["esquema"]
    sp = read_json(run / "splits.json")
    usables = set(features_utilizables(c, cargar_dev(run).columns))
    leak = read_json(run / "leakage_report.json") or {}
    viol_leak = {x["feature"] for x in leak.get("hallazgos", []) if x["clase"] == "violacion" and x.get("feature")}

    regs = [r for r in registros_vigentes(run, ctx) if not r.get("reproduccion_de")]
    if not regs:
        raise HarnessError("no hay candidatos vigentes.", "Corre baselines.py y experimentos.")
    C = []
    for r in regs:
        motivo = None
        malas = [f for f in r["features"] if f not in usables or f in viol_leak]
        if malas:
            motivo = f"usa features no permitidas por el contrato vigente: {malas}"
        elif rs.get("familias_permitidas") and r["familia"] not in rs["familias_permitidas"] and r["tipo"] != "baseline":
            motivo = f"familia {r['familia']} no permitida por restricciones"
        elif rs.get("max_segundos_entrenamiento") and (r["tiempos"].get("fit_s_max_fold") or 0) > float(rs["max_segundos_entrenamiento"]):
            motivo = f"entrenamiento {r['tiempos']['fit_s_max_fold']} s > máximo {rs['max_segundos_entrenamiento']} s"
        elif rs.get("max_ms_prediccion_1k") and (r["tiempos"].get("pred_ms_1k") or 0) > float(rs["max_ms_prediccion_1k"]):
            motivo = f"predicción {r['tiempos']['pred_ms_1k']} ms/1k > máximo {rs['max_ms_prediccion_1k']}"
        res = r["resumen"][pm]
        C.append({"run_id": r["run_id"], "id": r["id"], "tipo": r["tipo"], "nombre": r.get("nombre") or r["id"],
                  "familia": r["familia"], "n_features": len(r["features"]), "valor": res["valor"],
                  "std_entre_folds": res["std_entre_folds"], "std_entre_semillas": res["std_entre_semillas"],
                  "elegible": motivo is None, "motivo_no_elegible": motivo, "_rec": r})
    eleg = [x for x in C if x["elegible"]]
    if not eleg:
        raise HarnessError("ningún candidato es elegible.", "Revisa motivo_no_elegible en comparison.json y las restricciones del contrato.")
    signo = 1 if met.mayor_mejor else -1
    mejor = max(eleg, key=lambda x: signo * x["valor"])
    for x in C:
        x["deg"] = float(signo * (mejor["valor"] - x["valor"])) + 0.0

    # ---- inferencia
    advert = []
    metodo, inferencia, motivo_sin = None, True, None
    rep_por_fold = {f["id"]: f["repeticion"] for f in sp["folds"]}
    if esquema in ("kfold", "estratificado"):
        metodo = "nadeau_bengio"
        n_tr = np.mean([len(f["train"]) for f in sp["folds"]])
        n_va = np.mean([len(f["val"]) for f in sp["folds"]])
        ratio = n_va / n_tr
        J = len(sp["folds"])
        min_j = int(cmp_.get("min_folds_nb", 5))
        if J < min_j:
            inferencia, motivo_sin = False, f"solo {J} folds (Nadeau–Bengio requiere >= {min_j})"
        fold_tab = {x["run_id"]: pd.DataFrame(x["_rec"]["metricas_por_fold"]).groupby("fold")[pm].mean() for x in eleg}
        detalle = {"J": J, "n_val/n_train": round(float(ratio), 4), "supuestos": "folds intercambiables; corrección por solapamiento de entrenamientos"}
        unidades = J
        tq = stats.t.ppf(1 - alfa / 2, max(J - 1, 1))
        for x in eleg:
            # IC de la métrica del propio candidato (misma corrección de Nadeau–Bengio)
            f = fold_tab[x["run_id"]].values
            se = math.sqrt((1.0 / J + ratio) * float(np.var(f, ddof=1))) if J > 1 else float("nan")
            x["ic_metrica"] = [float(f.mean() - tq * se), float(f.mean() + tq * se)] if inferencia else None
            if x is mejor:
                continue
            d = signo * (fold_tab[mejor["run_id"]] - fold_tab[x["run_id"]]).reindex(fold_tab[mejor["run_id"]].index).values
            x.update(nadeau_bengio(d, ratio, margen, alfa))
    else:
        metodo = "bootstrap_bloques_temporales" if esquema == "temporal" else "bootstrap_entidades"
        L = 1
        if esquema == "temporal":
            L = int(cmp_.get("bloque_periodos") or math.ceil(math.sqrt(int(c["validacion"]["tamano_validacion"]))))
        B = int(cmp_.get("n_boot", 1000))
        preds = {x["run_id"]: read_table(run / x["_rec"]["preds_ref"]) for x in eleg}
        bs = Bootstrap(preds[mejor["run_id"]], esquema, L, B, rep_por_fold)
        unidades = bs.n_unidades
        detalle = {"unidades": unidades, "largo_bloque": L if esquema == "temporal" else None, "B": B,
                   "unidad": "bloque de periodos (todas las entidades)" if esquema == "temporal" else "entidad"}
        if unidades < int(cmp_.get("min_unidades", 20)):
            inferencia, motivo_sin = False, f"solo {unidades} unidades de remuestreo (mínimo {cmp_.get('min_unidades', 20)})"
        else:
            clases = mejor["_rec"].get("clases")
            agg = m.get("agregacion", "media_folds")
            dist_m = bs.distribucion(preds[mejor["run_id"]], pm, clases, agg)
            for x in eleg:
                dist = dist_m if x is mejor else bs.distribucion(preds[x["run_id"]], pm, clases, agg)
                df_ = dist[np.isfinite(dist)]
                # IC de la métrica del propio candidato, con la unidad de remuestreo del diseño (entidad o bloque)
                x["ic_metrica"] = [float(np.quantile(df_, alfa / 2)), float(np.quantile(df_, 1 - alfa / 2))]
                if x is mejor:
                    continue
                dd = signo * (dist_m - dist)
                dd = dd[np.isfinite(dd)]
                x.update({"ic": [float(np.quantile(dd, alfa / 2)), float(np.quantile(dd, 1 - alfa / 2))],
                          "p_noinf": float((1 + np.sum(dd >= margen)) / (len(dd) + 1)),
                          "p_inf": float((1 + np.sum(dd <= margen)) / (len(dd) + 1))})
    mejor.update({"ic": [0.0, 0.0], "veredicto": "mejor"})
    for x in C:
        x.setdefault("ic_metrica", None)
    otros = [x for x in eleg if x is not mejor]
    if inferencia and otros:
        a1, a2 = holm([x["p_noinf"] for x in otros]), holm([x["p_inf"] for x in otros])
        for x, q1, q2 in zip(otros, a1, a2):
            x["p_noinf_holm"], x["p_inf_holm"] = q1, q2
            x["veredicto"] = "no_inferior" if q1 < alfa else ("inferior" if q2 < alfa else "inconcluso")
    else:
        for x in otros:
            x["veredicto"] = "sin_inferencia"

    # ---- elección
    clave = lambda x: (_rango(x["familia"], orden), x["n_features"], x["deg"])
    if override:
        cand = [x for x in C if x["id"] == override or x["run_id"] == override]
        if not cand:
            raise HarnessError(f"--override {override}: no es un candidato vigente.", "Usa el id de un candidato listado en comparison.json.")
        sel = cand[-1]
        if not sel["elegible"]:
            raise HarnessError(f"--override {override}: el candidato no es elegible ({sel['motivo_no_elegible']}).",
                               "Un override no permite saltarse leakage comprobado, evaluaciones inválidas ni restricciones.")
        dec = (run / "decisions.md").read_text(encoding="utf-8") if (run / "decisions.md").exists() else ""
        if not re.search(r"^## D-\d+.*$[\s\S]*?" + re.escape(sel["id"]), dec, flags=re.M) or len((justificacion or "").strip()) < 30:
            raise HarnessError(f"--override {override} sin justificación registrada.",
                               f"Agrega una entrada `## D-NNN` en decisions.md que mencione {sel['id']} y explique por qué, y pasa --justificacion (>= 30 caracteres).")
        estado, criterio = "override", f"override manual: {justificacion}"
    elif inferencia:
        noinf = [x for x in eleg if x["veredicto"] in ("mejor", "no_inferior")]
        sel = min(noinf, key=clave)
        mas_simples_inc = [x for x in eleg if clave(x) < clave(sel) and x["veredicto"] == "inconcluso" and x["deg"] <= margen]
        if mas_simples_inc:
            estado = "provisional"
            if cmp_.get("desempate", "mas_simple") == "mas_simple":
                sel = min(mas_simples_inc, key=clave)
            criterio = (f"la comparación con {[x['run_id'] for x in mas_simples_inc]} (más simples) es inconclusa; "
                        f"se aplicó el desempate predefinido '{cmp_.get('desempate', 'mas_simple')}'")
        else:
            estado = "concluyente"
            criterio = (f"el más simple entre los no inferiores al mejor ({mejor['run_id']}) con margen {margen} "
                        f"y alfa {alfa} (Holm)")
    else:
        pool = [x for x in eleg if x["deg"] <= margen]
        sel = min(pool, key=clave)
        estado = "operativa"
        criterio = f"selección operativa SIN inferencia válida ({motivo_sin}): el más simple con degradación puntual <= margen"

    # ---- contra el ingenuo
    ing = [x for x in eleg if x["familia"] == "ingenuo"]
    vs_ing = None
    if ing:
        mi = max(ing, key=lambda x: signo * x["valor"])
        mejora = signo * (sel["valor"] - mi["valor"])
        rel = mejora / abs(mi["valor"]) if mi["valor"] else None
        vs_ing = {"ingenuo": mi["run_id"], "valor_ingenuo": mi["valor"], "mejora": float(mejora),
                  "mejora_relativa": None if rel is None else float(rel)}
        if sel is not mi and inferencia:
            if metodo == "nadeau_bengio":
                ft = {k: pd.DataFrame(x["_rec"]["metricas_por_fold"]).groupby("fold")[pm].mean() for k, x in (("s", sel), ("i", mi))}
                # d = mejora del elegido sobre el ingenuo; H0: d <= 0  => p unilateral = p_inf con margen 0
                r = nadeau_bengio(signo * (ft["s"] - ft["i"]).values, ratio, 0.0, alfa)
                vs_ing.update({"ic_mejora": [float(v) for v in r["ic"]], "p_no_mejora": r["p_inf"]})
            else:
                dd = signo * (bs.distribucion(preds[sel["run_id"]], pm, mejor["_rec"].get("clases"), m.get("agregacion", "media_folds"))
                              - bs.distribucion(preds[mi["run_id"]], pm, mejor["_rec"].get("clases"), m.get("agregacion", "media_folds")))
                vs_ing.update({"ic_mejora": [float(np.quantile(dd, alfa / 2)), float(np.quantile(dd, 1 - alfa / 2))],
                               "p_no_mejora": float((1 + np.sum(dd <= 0)) / (len(dd) + 1))})
            if vs_ing.get("p_no_mejora", 1) >= alfa:
                advert.append(f"el elegido ({sel['run_id']}) NO supera al mejor ingenuo ({mi['run_id']}) con evidencia suficiente.")
        if rel is not None and rel > 0.9:
            advert.append(f"mejora de {rel:.0%} sobre el ingenuo: sospechosamente buena; asume leakage hasta demostrar lo contrario.")
    n_exp = len([x for x in C if x["tipo"] == "experimento"])
    advert.append(f"{n_exp} experimentos comparados sobre los mismos folds: la estimación del elegido tiene sesgo optimista de selección; "
                  "Holm solo controla la familia 'vs mejor'. El holdout es la estimación confirmatoria.")
    if not inferencia:
        advert.append(f"sin inferencia válida: {motivo_sin}.")

    r = sel["_rec"]
    congelado = {"run_id": r["run_id"], "id": r["id"], "tipo": r["tipo"], "nombre": sel["nombre"], "codigo_sha256": r["codigo_sha256"],
                 "codigo_ref": r["codigo_ref"], "features": r["features"], "familia": r["familia"], "config": r["config"],
                 "semillas": r["semillas"], "estocastico": r["estocastico"], "umbral": c["metrica"].get("umbral"),
                 "eval_hash": ctx["eval_hash"]}
    out = {"fecha": ahora(), "eval_hash": ctx["eval_hash"], "splits_hash": ctx["splits_hash"],
           "experiments_sha256": sha256_file(run / "experiments.jsonl"), "metrica": pm, "mayor_mejor": met.mayor_mejor,
           "agregacion": m.get("agregacion", "media_folds"), "margen_practico": margen, "alfa": alfa,
           "metodo": metodo, "detalle_metodo": detalle, "inferencia_valida": inferencia, "motivo_sin_inferencia": motivo_sin,
           "mejor": mejor["run_id"],
           "candidatos": [{k: v for k, v in x.items() if not k.startswith("_")} for x in C],
           "seleccion": {"run_id": sel["run_id"], "id": sel["id"], "estado": estado, "criterio": criterio},
           "vs_ingenuo": vs_ing, "congelado": congelado, "advertencias": advert,
           "familia_holm": "cada candidato elegible vs el mejor (no inferioridad e inferioridad por separado)",
           "nota_ic": (f"IC bilaterales al {1 - alfa:.0%}. 'IC {pm}' es la incertidumbre de la métrica de cada candidato con la unidad "
                       "de remuestreo del diseño; la desviación entre folds puede subestimarla (p. ej. con folds estratificados). "
                       "Las decisiones usan pruebas unilaterales sobre la degradación vs el mejor, con Holm.")}
    return out


def _md(out: dict) -> str:
    pm = out["metrica"]
    L = [f"# Comparación de modelos", "",
         f"Métrica: **{pm}** ({'mayor' if out['mayor_mejor'] else 'menor'} es mejor), agregación {out['agregacion']}, "
         f"margen práctico {out['margen_practico']}, alfa {out['alfa']}.",
         f"Método: {out['metodo']} {out['detalle_metodo']}. Inferencia válida: {out['inferencia_valida']}"
         + (f" ({out['motivo_sin_inferencia']})" if out["motivo_sin_inferencia"] else "") + ".", "",
         "| candidato | tipo | familia | #feat | " + pm + " | IC " + pm + " | std folds | std semillas | degradación | IC degradación | veredicto | elegible |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    f = lambda v: "—" if v is None else (f"{v:.4g}" if isinstance(v, (int, float)) else str(v))
    for x in sorted(out["candidatos"], key=lambda x: x["deg"]):
        ic = x.get("ic")
        im = x.get("ic_metrica")
        L.append(f"| {x['run_id']} {x['nombre']} | {x['tipo']} | {x['familia']} | {x['n_features']} | {f(x['valor'])} | "
                 f"{'—' if not im else f'[{im[0]:.4g}, {im[1]:.4g}]'} | "
                 f"{f(x['std_entre_folds'])} | {f(x['std_entre_semillas'])} | {f(x['deg'])} | "
                 f"{'—' if not ic else f'[{ic[0]:.4g}, {ic[1]:.4g}]'} | {x.get('veredicto', '—')} | "
                 f"{'sí' if x['elegible'] else 'no: ' + str(x['motivo_no_elegible'])} |")
    s = out["seleccion"]
    L += ["", f"**Elegido: {s['run_id']}** — estado `{s['estado']}`. Criterio: {s['criterio']}.", ""]
    if out["vs_ingenuo"]:
        L.append(f"Contra el mejor ingenuo: {out['vs_ingenuo']}.")
    for a in out["advertencias"]:
        L.append(f"- AVISO: {a}")
    L.append(f"\n{out['nota_ic']}")
    return "\n".join(L)


def criterios(run):
    """Criterios de cierre de la fase de experimentación."""
    from contract import cargar_contrato, contexto_eval
    from experiment_log import bitacora, normalizar, registros_vigentes
    run = Path(run)
    P = []
    c = cargar_contrato(run)
    ctx = contexto_eval(run)
    out = read_json(run / "comparison.json")
    if out is None:
        return [("falta comparison.json.", f"Corre `python scripts/compare_models.py --run {run}`.")], []
    if out["experiments_sha256"] != sha256_file(run / "experiments.jsonl") or out["eval_hash"] != ctx["eval_hash"]:
        P.append(("comparison.json está desactualizado respecto de experiments.jsonl o del contrato.", "Vuelve a correr compare_models.py."))
    exps = [r for r in registros_vigentes(run, ctx) if r["tipo"] == "experimento" and not r.get("reproduccion_de")]
    textos, firmas, distintas, repetidas = set(), set(), [], []
    for r in exps:
        t, fz = normalizar(r["hipotesis"]["hipotesis"]), r["firma_config"]
        if t in textos or fz in firmas:
            repetidas.append(r["id"])
        else:
            distintas.append(r["id"])
        textos.add(t)
        firmas.add(fz)
    obj = int((c.get("experimentos") or {}).get("objetivo_hipotesis", 5))
    todo = bitacora(run)
    cierre = [r for r in todo if r.get("tipo") == "cierre" and r.get("eval_hash") == ctx["eval_hash"]]
    if len(distintas) < obj and not cierre:
        P.append((f"hay {len(distintas)} hipótesis distintas ejecutadas (objetivo {obj})"
                  + (f"; no cuentan por repetidas: {repetidas}" if repetidas else "") + ".",
                  "Ejecuta más hipótesis genuinamente distintas, o cierra con `experiment_log.py stop --motivo ... --evidencia ...` (lo evalúa el revisor)."))
    if cierre and not exps:
        P.append(("cierre anticipado sin ningún experimento de hipótesis.", "Ejecuta al menos un experimento."))
    concl = {r["id"] for r in todo if r.get("tipo") == "conclusion"}
    sin = sorted({r["id"] for r in exps} - concl)
    if sin:
        P.append((f"experimentos sin resultado observado registrado: {sin}.",
                  "Registra cada uno con `experiment_log.py conclude <id> --resultado ... --decision ...`."))
    dec = (run / "decisions.md").read_text(encoding="utf-8") if (run / "decisions.md").exists() else ""
    if not re.search(r"^## D-\d+", dec, flags=re.M):
        P.append(("decisions.md no tiene ninguna decisión registrada.", "Agrega entradas `## D-NNN — título` con su justificación."))
    return P, ["experiments.jsonl", "comparison.json"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--override")
    ap.add_argument("--justificacion")
    a = ap.parse_args()
    from gate import requerir
    requerir(a.run, "experimentos")
    out = comparar(a.run, a.override, a.justificacion)
    atomic_write_json(Path(a.run) / "comparison.json", out)
    md = _md(out)
    atomic_write_text(Path(a.run) / "comparison.md", md + "\n")
    print(md)
    return 0


if __name__ == "__main__":
    cli(main)
