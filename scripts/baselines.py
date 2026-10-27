"""
Fase 2 — Escalera de baselines.

Uso:
    python scripts/baselines.py --run runs/<p>
    python scripts/gate.py advance baselines --run runs/<p>

Corre, con el mismo runner y los mismos folds que cualquier experimento:
  regression      B01 media, B02 mediana (ingenuos) · B10 Ridge (lineal) · B20 HistGradientBoosting (gbm)
  classification  B01 prior de clases (ingenuo) · B10 regresión logística (lineal) · B20 HistGradientBoosting (gbm)
  forecasting     B01 media global, B02 último valor por entidad, B03 media por entidad,
                  B04 estacional ingenuo (si hay estacionalidad) · B10 Ridge · B20 HistGradientBoosting
Los modelos lineal y GBM usan todas las features que el contrato permite, con todo el
preprocesamiento dentro del pipeline. HistGradientBoosting sin early stopping es determinista,
así que se corre con una sola semilla.

Escribe baselines.json. Sin baselines aprobados, experiment_log.py no permite experimentos.
"""
from __future__ import annotations

import argparse
import inspect
from pathlib import Path

from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.pipeline import Pipeline
import numpy as np

from common import HarnessError, atomic_write_json, cli, read_json, sha256_file, sha256_text
from modelos import EstacionalIngenuo, MediaEntidad, NaiveUltimo, preprocesador_arboles, preprocesador_lineal

FUENTE = Path(__file__).read_text(encoding="utf-8")


def _gbm_reg(config, seed):
    return Pipeline([("pre", preprocesador_arboles()),
                     ("modelo", HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, early_stopping=False, random_state=seed))])


def _gbm_clf(config, seed):
    return Pipeline([("pre", preprocesador_arboles()),
                     ("modelo", HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, early_stopping=False, random_state=seed))])


def _ridge(config, seed):
    return Pipeline([("pre", preprocesador_lineal()), ("modelo", RidgeCV(alphas=np.logspace(-3, 3, 13)))])


def _logit(config, seed):
    return Pipeline([("pre", preprocesador_lineal()), ("modelo", LogisticRegression(C=1.0, max_iter=2000))])


def escalera(c: dict, feats: list[str]) -> list[dict]:
    """Define los baselines aplicables al contrato. Cada uno: id, nombre, familia, features, build."""
    pt = c["problem_type"]
    B = []

    def add(id_, nombre, familia, build, features, requerido=True):
        B.append({"id": id_, "nombre": nombre, "familia": familia, "build": build, "features": features,
                  "requerido": requerido})

    if pt == "classification":
        add("B01", "prior_de_clases", "ingenuo", lambda cf, s: DummyClassifier(strategy="prior"), feats[:1])
        add("B10", "logistica", "lineal", _logit, feats)
        add("B20", "hgb", "gbm", _gbm_clf, feats)
        return B
    add("B01", "media", "ingenuo", lambda cf, s: DummyRegressor(strategy="mean"), feats[:1])
    if pt == "regression":
        add("B02", "mediana", "ingenuo", lambda cf, s: DummyRegressor(strategy="median"), feats[:1])
    else:
        add("B02", "ultimo_valor", "ingenuo", lambda cf, s: NaiveUltimo(), [])
        add("B03", "media_entidad", "ingenuo", lambda cf, s: MediaEntidad(), [])
        m = c["prediccion"].get("estacionalidad")
        if m:
            add("B04", "estacional_ingenuo", "ingenuo", lambda cf, s, m=int(m): EstacionalIngenuo(m=m), [])
    add("B10", "ridge", "lineal", _ridge, feats)
    add("B20", "hgb", "gbm", _gbm_reg, feats)
    return B


def correr(run) -> dict:
    from contract import cargar_contrato, cargar_dev, contexto_eval, features_utilizables
    from experiment_log import ejecutar, registros_vigentes
    from gate import requerir
    run = Path(run)
    requerir(run, "baselines")
    c = cargar_contrato(run)
    ctx = contexto_eval(run)
    feats = features_utilizables(c, cargar_dev(run).columns)
    if not feats:
        raise HarnessError("no hay features utilizables según el contrato.", "Revisa `features` y `columnas_excluidas`.")
    previos = {r["id"]: r for r in registros_vigentes(run, ctx) if r["tipo"] == "baseline"}
    resumen = []
    for b in escalera(c, feats):
        codigo = FUENTE + f"\n# baseline {b['id']} ({b['nombre']}): " + inspect.getsource(b["build"]).strip()
        prev = previos.get(b["id"])
        if prev and prev["features"] == b["features"] and prev["codigo_sha256"] == sha256_text(codigo):
            rec = prev   # ya corrido con el mismo código y features: no se repite
        else:
            spec = {"id": b["id"], "tipo": "baseline", "nombre": b["nombre"], "hipotesis": None, "features": b["features"],
                    "familia": b["familia"], "estocastico": False, "config": {}, "build": b["build"], "codigo": codigo,
                    "contexto_ok": True}
            rec = ejecutar(run, spec)
        pm = c["metrica"]["principal"]
        val = rec.get("resumen", {}).get(pm, {})
        resumen.append({"id": b["id"], "nombre": b["nombre"], "familia": b["familia"], "run_id": rec["run_id"],
                        "estado": rec["estado"], "error": rec.get("error"), "requerido": b["requerido"],
                        pm: val.get("valor"), "std_entre_folds": val.get("std_entre_folds")})
        print(f"{b['id']} {b['nombre']:<20} {rec['estado']:<8} {pm}={val.get('valor')}")
    out = {"eval_hash": ctx["eval_hash"], "splits_hash": ctx["splits_hash"], "metrica": c["metrica"]["principal"],
           "baselines": resumen, "experiments_sha256": sha256_file(run / "experiments.jsonl")}
    atomic_write_json(run / "baselines.json", out)
    return out


def criterios(run):
    from contract import contexto_eval
    run = Path(run)
    B = read_json(run / "baselines.json")
    if B is None:
        return [("falta baselines.json.", f"Corre `python scripts/baselines.py --run {run}`.")], []
    P = []
    ctx = contexto_eval(run)
    if (B["eval_hash"], B["splits_hash"]) != (ctx["eval_hash"], ctx["splits_hash"]):
        P.append(("baselines.json corresponde a otro contrato/splits.", "Vuelve a correr baselines.py."))
    fall = [b["id"] for b in B["baselines"] if b["requerido"] and b["estado"] != "ok"]
    if fall:
        P.append((f"baselines fallidos: {fall}.", "Revisa el error en experiments.jsonl (campo error/traceback) y corrige datos o contrato."))
    fams = {b["familia"] for b in B["baselines"] if b["estado"] == "ok"}
    for f in ("ingenuo", "lineal", "gbm"):
        if f not in fams:
            P.append((f"la escalera no tiene un baseline '{f}' válido.", "Vuelve a correr baselines.py."))
    return P, ["baselines.json"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    a = ap.parse_args()
    out = correr(a.run)
    fall = [b["id"] for b in out["baselines"] if b["estado"] != "ok"]
    if fall:
        print(f"BLOQUEADO: baselines fallidos {fall}. Cómo resolverlo: revisa `error` en experiments.jsonl.")
        return 1
    print(f"OK: escalera de baselines en baselines.json. Siguiente: `python scripts/gate.py advance baselines --run {a.run}`.")
    return 0


if __name__ == "__main__":
    cli(main)
