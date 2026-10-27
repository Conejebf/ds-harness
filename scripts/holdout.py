"""
Fase 6 — Evaluación ÚNICA del holdout congelado.

Uso:
    python scripts/holdout.py evaluate --run runs/<p> [--recuperar]
    python scripts/holdout.py status   --run runs/<p>

Ciclo de vida (state.json → holdout.estado):
    sellado ──evaluate──> en_curso ──(éxito)──> evaluado
                              └─(falla)─ queda en_curso; `--recuperar` reintenta con EL MISMO candidato congelado.

Antes de abrir el holdout se escribe holdout_freeze.json: candidato (código exacto por hash),
features, hiperparámetros, semillas, política de umbral, métricas y baseline ingenuo de referencia.
Después no se puede cambiar nada de eso.

Procedimiento (el mismo aprobado en CV):
  * no temporal: el candidato se re-entrena con TODO desarrollo y predice el holdout;
  * temporal (forecasting): origen móvil con ventanas de `validacion.tamano_validacion`; cada ventana
    se predice con el candidato re-entrenado con todo lo observado hasta su origen menos el gap
    (desarrollo + ventanas anteriores del holdout). Ningún target posterior al origen se usa.
El resultado lleva IC por bootstrap con la misma unidad de remuestreo que la comparación.

Reglas:
  * Repetir `evaluate` devuelve el resultado existente; no recalcula.
  * Un resultado decepcionante NO autoriza probar otro modelo contra el mismo holdout: si cambia el
    candidato, `evaluate` se niega. Una nueva iteración exige `contract.py validate --new-version`, y
    el holdout queda marcado como ya observado (deja de ser una estimación limpia).
Límite: aislamiento operativo, no barrera de seguridad (ver contract.py).
"""
from __future__ import annotations

import argparse
import math
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from common import (HarnessError, ahora, atomic_write_json, cli, find_table, load_state, log_event, read_json, read_table,
                    sha256_file, sha256_json, state_tx, write_table)


def _congelar(run: Path) -> dict:
    cmpj = read_json(run / "comparison.json")
    from contract import cargar_contrato
    c = cargar_contrato(run)
    fz = {"candidato": cmpj["congelado"], "comparison_sha256": sha256_file(run / "comparison.json"),
          "contract_sha256": sha256_file(run / "contract.yaml"), "metricas": [c["metrica"]["principal"], *(c["metrica"].get("secundarias") or [])],
          "agregacion": c["metrica"].get("agregacion", "media_folds"), "umbral": c["metrica"].get("umbral"),
          "referencia_ingenuo": (cmpj.get("vs_ingenuo") or {}).get("ingenuo"), "seleccion": cmpj["seleccion"]}
    fz["hash"] = sha256_json(fz)
    return fz


def _ventanas(dev: pd.DataFrame, hold: pd.DataFrame, c: dict) -> list[dict]:
    """Define las ventanas de evaluación (filas de entrenamiento y de prueba por ventana)."""
    if "_t" in hold and c["validacion"]["esquema"] == "temporal":
        L = int(c["validacion"]["tamano_validacion"])
        gap = int(c["validacion"].get("gap", 0) or 0)
        t0, t1 = int(hold["_t"].min()), int(hold["_t"].max())
        todo = pd.concat([dev, hold], ignore_index=True)
        V = []
        for k, ini in enumerate(range(t0, t1 + 1, L)):
            fin = min(ini + L - 1, t1)
            V.append({"id": f"h{k}", "origen": ini - 1, "train": todo[todo["_t"] <= ini - gap - 1],
                      "test": hold[(hold["_t"] >= ini) & (hold["_t"] <= fin)]})
        return V
    return [{"id": "h0", "origen": None, "train": dev, "test": hold}]


def _predecir(spec, rec, c, V, clases, umbral) -> pd.DataFrame:
    from experiment_log import ajustar_predecir
    idc, target = c["datos"]["id_fila"], c["datos"]["target"]
    out = []
    for v in V:
        for s in rec["semillas"]:
            r = ajustar_predecir(spec, c, v["train"], v["test"], s, clases, umbral)
            pf = pd.DataFrame({"id": v["test"][idc].astype(str).values, "fold": v["id"], "repeticion": 0, "semilla": s,
                               "y_true": v["test"][target].values, "y_pred": r["yhat"]})
            if r["P"] is not None:
                for j, k in enumerate(clases):
                    pf[f"p__{k}"] = r["P"][:, j]
            if "_entidad" in v["test"]:
                pf["entidad"] = v["test"]["_entidad"].values
            if v["origen"] is not None:
                pf["t"] = v["test"]["_t"].values
                pf["origen"] = v["origen"]
                pf["horizonte"] = pf["t"] - v["origen"]
            out.append(pf)
    return pd.concat(out, ignore_index=True)


def _metricas(preds, metricas, clases, agregacion) -> dict:
    from metrics import calcular
    out = {}
    for k in metricas:
        por_s = {}
        for s, g in preds.groupby("semilla"):
            if agregacion == "pooled" or g["fold"].nunique() == 1:
                P = g[[f"p__{c}" for c in clases]].values if clases and f"p__{clases[0]}" in g else None
                por_s[int(s)] = calcular(k, g["y_true"].values, g["y_pred"].values, P, clases)
            else:
                vals = []
                for _, gf in g.groupby("fold"):
                    P = gf[[f"p__{c}" for c in clases]].values if clases and f"p__{clases[0]}" in gf else None
                    vals.append(calcular(k, gf["y_true"].values, gf["y_pred"].values, P, clases))
                por_s[int(s)] = float(np.mean(vals))
        out[k] = {"valor": float(np.mean(list(por_s.values()))), "por_semilla": por_s}
    return out


def evaluar(run, recuperar: bool = False) -> dict:
    from compare_models import Bootstrap
    from contract import cargar_contrato, cargar_dev
    from experiment_log import bitacora, contexto_cv, spec_desde_registro
    from gate import registrar, requerir
    from metrics import METRICAS
    run = Path(run)
    st = load_state(run)
    ho = st["holdout"]
    fz_actual = _congelar(run) if (run / "comparison.json").exists() else None
    if ho["estado"] == "evaluado":
        if fz_actual and ho.get("congelado") == fz_actual["hash"]:
            R = read_json(run / "holdout_result.json")
            print("El holdout ya se evaluó para este candidato; se devuelve el resultado existente (no se recalcula).")
            return R
        raise HarnessError(
            f"el holdout ya se evaluó con otro candidato ({(ho.get('candidato') or {}).get('run_id')}).",
            "No se prueba otro modelo contra el mismo holdout. Para una nueva iteración: `contract.py validate --new-version` "
            "(el holdout quedará marcado como observado; lo ideal es conseguir datos nuevos para un holdout limpio).")
    requerir(run, "holdout")
    if ho["estado"] == "en_curso":
        if not recuperar:
            raise HarnessError("una evaluación anterior del holdout quedó interrumpida.",
                               "Usa `holdout.py evaluate --recuperar` (solo con el mismo candidato congelado; queda registrado).")
        if ho.get("congelado") != fz_actual["hash"]:
            raise HarnessError("la recuperación no puede cambiar el candidato congelado.",
                               "Restaura comparison.json/contrato al estado congelado, o declara una nueva versión.")
    elif ho["estado"] != "sellado":
        raise HarnessError(f"estado del holdout inesperado: {ho['estado']}.", "Revisa state.json; re-valida el contrato.")
    fz = fz_actual
    atomic_write_json(run / "holdout_freeze.json", fz)
    with state_tx(run) as s2:
        s2["holdout"].update({"estado": "en_curso", "congelado": fz["hash"], "candidato": fz["candidato"]})
        s2["holdout"].setdefault("eventos", []).append({"fecha": ahora(), "tipo": "recuperacion" if recuperar else "apertura",
                                                         "candidato": fz["candidato"]["run_id"]})
        log_event(s2, "holdout_abierto", candidato=fz["candidato"]["run_id"], recuperacion=recuperar)

    try:
        c = cargar_contrato(run)
        man = read_json(run / "holdout_manifest.json")
        hp = find_table(run / "_sealed" / "holdout")
        if hp is None or sha256_file(hp) != man["sha256_holdout"]:
            raise RuntimeError("el archivo sellado del holdout no existe o no coincide con su hash del manifiesto")
        hold = read_table(hp)
        dev = cargar_dev(run)
        K = contexto_cv(run)
        regs = {r["run_id"]: r for r in bitacora(run) if r.get("tipo") in ("baseline", "experimento")}
        rec = regs[fz["candidato"]["run_id"]]
        if rec["codigo_sha256"] != fz["candidato"]["codigo_sha256"]:
            raise RuntimeError("el código del candidato no coincide con el congelado")
        V = _ventanas(dev, hold, c)
        pm = c["metrica"]["principal"]
        preds = _predecir(spec_desde_registro(run, rec), rec, c, V, K["clases"], K["umbral"])
        M = _metricas(preds, fz["metricas"], K["clases"], fz["agregacion"])
        esquema = c["validacion"]["esquema"]
        L = 1
        if esquema == "temporal":
            cmp_ = c.get("comparacion") or {}
            L = int(cmp_.get("bloque_periodos") or math.ceil(math.sqrt(int(c["validacion"]["tamano_validacion"]))))
        alfa = float(c["comparacion"].get("alfa", 0.05))
        B = int(c["comparacion"].get("n_boot", 1000))
        bs = Bootstrap(preds, esquema if esquema in ("temporal", "grupos") else "filas", L, B, {v["id"]: 0 for v in V})
        dist = bs.distribucion(preds, pm, K["clases"], fz["agregacion"])
        ic = [float(np.nanquantile(dist, alfa / 2)), float(np.nanquantile(dist, 1 - alfa / 2))]
        ref = None
        if fz["referencia_ingenuo"] and fz["referencia_ingenuo"] != rec["run_id"]:
            rr = regs[fz["referencia_ingenuo"]]
            pr = _predecir(spec_desde_registro(run, rr), rr, c, V, K["clases"], K["umbral"])
            Mr = _metricas(pr, [pm], K["clases"], fz["agregacion"])
            dr = bs.distribucion(pr, pm, K["clases"], fz["agregacion"])
            signo = 1 if METRICAS[pm].mayor_mejor else -1
            dd = signo * (dist - dr)
            ref = {"run_id": rr["run_id"], "valor": Mr[pm]["valor"], "mejora": float(signo * (M[pm]["valor"] - Mr[pm]["valor"])),
                   "ic_mejora": [float(np.nanquantile(dd, alfa / 2)), float(np.nanquantile(dd, 1 - alfa / 2))],
                   "p_no_mejora": float((1 + np.sum(dd <= 0)) / (len(dd) + 1))}
        cv = rec["resumen"][pm]["valor"]
        pp = write_table(preds, run / "holdout_preds")
        R = {"fecha": ahora(), "version_estudio": st.get("version_estudio", 1), "candidato": fz["candidato"],
             "freeze_sha256": sha256_file(run / "holdout_freeze.json"), "metricas": M, "metrica_principal": pm,
             "ic_principal": ic, "unidades_bootstrap": bs.n_unidades, "referencia_ingenuo": ref,
             "cv_principal": cv, "diferencia_vs_cv": float(M[pm]["valor"] - cv), "n_filas": int(len(hold)),
             "ventanas": [{"id": v["id"], "origen": v["origen"], "n_train": int(len(v["train"])), "n_test": int(len(v["test"]))} for v in V],
             "procedimiento": ("origen móvil: cada ventana re-entrena con lo observado hasta su origen (menos gap)"
                               if V[0]["origen"] is not None else "re-entrenado con todo desarrollo; una sola evaluación"),
             "observaciones_previas": ho.get("observaciones_previas", []),
             "preds_ref": pp.relative_to(run).as_posix()}
        atomic_write_json(run / "holdout_result.json", R)
    except Exception as e:
        with state_tx(run) as s2:
            s2["holdout"]["eventos"].append({"fecha": ahora(), "tipo": "error", "error": f"{type(e).__name__}: {e}"})
        raise HarnessError(f"la evaluación del holdout falló ({type(e).__name__}: {e}); el holdout quedó en estado en_curso.\n"
                           + traceback.format_exc()[-1500:],
                           "Corrige la causa SIN cambiar el candidato y corre `holdout.py evaluate --recuperar`.", codigo=1)
    with state_tx(run) as s2:
        s2["holdout"].update({"estado": "evaluado", "resultado_sha256": sha256_file(run / "holdout_result.json")})
        s2["holdout"]["eventos"].append({"fecha": ahora(), "tipo": "evaluado", "candidato": fz["candidato"]["run_id"]})
        log_event(s2, "holdout_evaluado", candidato=fz["candidato"]["run_id"])
    registrar(run, "holdout", ["holdout_freeze.json", "holdout_result.json", R["preds_ref"]])
    return R


def criterios(run):
    run = Path(run)
    st = load_state(run)
    if st["holdout"]["estado"] != "evaluado" or not (run / "holdout_result.json").exists():
        return [("el holdout no se ha evaluado.", f"Corre `python scripts/holdout.py evaluate --run {run}`.")], []
    R = read_json(run / "holdout_result.json")
    return [], ["holdout_freeze.json", "holdout_result.json", R["preds_ref"]]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("accion", choices=["evaluate", "status"])
    ap.add_argument("--run", required=True)
    ap.add_argument("--recuperar", action="store_true")
    a = ap.parse_args()
    if a.accion == "status":
        ho = load_state(a.run)["holdout"]
        print(f"estado={ho['estado']} candidato={(ho.get('candidato') or {}).get('run_id')} observaciones_previas={ho.get('observaciones_previas')}")
        return 0
    R = evaluar(a.run, a.recuperar)
    pm = R["metrica_principal"]
    print(f"Holdout ({R['n_filas']} filas): {pm} = {R['metricas'][pm]['valor']:.5g} IC [{R['ic_principal'][0]:.5g}, {R['ic_principal'][1]:.5g}]; "
          f"CV = {R['cv_principal']:.5g}. Referencia ingenua: {R['referencia_ingenuo']}")
    return 0


if __name__ == "__main__":
    cli(main)
