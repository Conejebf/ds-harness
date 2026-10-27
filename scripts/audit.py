"""
Fase 5 — Auditoría determinista (la corre el revisor adversarial; también se puede correr a mano).

Uso:
    python scripts/audit.py --run runs/<p> [--alcance seleccionado|todos|ninguno]

Verificaciones (cada una: verificado | falla | no_verificable):
  A-INTEGRIDAD     fases previas válidas; hashes de predicciones y de código congelado.
  A-SPLITS         los splits se reconstruyen idénticos desde el contrato; sin solapamientos ni
                   gaps violados; holdout disjunto de desarrollo (por fila, entidad o periodo).
  A-METRICAS       las métricas registradas se reproducen desde las predicciones guardadas.
  A-REEJECUCION    re-entrena según el alcance (contrato: auditoria.alcance_reejecucion) y compara
                   métricas por fold y semilla con tolerancias explícitas (no igualdad bit a bit).
  A-AISLAMIENTO    (estático) el código de los experimentos no carga datos ni toca _sealed/;
                   (dinámico) la predicción de una fila no depende de las demás filas del lote
                   (detecta transformaciones que se "ajustan" al predecir).
  A-DISP-<f>       disponibilidad y alineación temporal de cada feature del modelo elegido:
                   desfase exacto vs target (prueba determinista), reconstrucción por linaje
                   declarado ("viaje en el tiempo") y asociación con el target futuro (señal).
                   Sin linaje ni timestamps suficientes => no_verificable (no se afirma lo contrario).
  A-SENAL-<f>      (importante) feature sin linaje verificable que se asocia más con el target futuro.
  A-LEAKAGE        hallazgos de leakage resueltos; señales justificadas que el revisor debe evaluar.
  A-SOSPECHOSO     mejora sospechosamente grande sobre el ingenuo.
  A-HOLDOUT        el holdout no se abrió antes de la revisión y ningún código lo referencia.
  A-BUSQUEDA       intensidad de búsqueda vs número de folds (sesgo de selección).
  A-CIERRE         si hubo cierre anticipado de la exploración, se marca para revisión.

Escribe audit.json (con el hash del conjunto de artefactos auditados). Sale con 1 si hay una
falla bloqueante. Una feature del futuro disfrazada (con ruido o transformada) y sin linaje NO
siempre es detectable: ese caso se reporta como no_verificable + señal estadística.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from common import (FASES, HarnessError, ahora, atomic_write_json, cli, find_table, load_state, read_json, read_table, sha256_text,
                    sha256_file, sha256_json)

PATRON_DATOS = re.compile(r"(read_csv|read_parquet|read_pickle|read_table|read_excel|np\.load|open\(|_sealed|holdout|dev\.parquet|dev\.pkl|splits\.json)")


def _chk(id_, nombre, veredicto, evidencia, severidad=None, accion=None, artefacto=None):
    return {"id": id_, "nombre": nombre, "veredicto": veredicto, "evidencia": evidencia,
            "severidad": severidad if veredicto == "falla" else None, "accion": accion, "artefacto": artefacto}


def conjunto_revisado(run) -> dict:
    """Hashes del conjunto de artefactos que audita/revisa la fase 5 (fases contrato..diagnostico + decisiones)."""
    st = load_state(run)
    H = {}
    for f in FASES[: FASES.index("revision")]:
        H.update((st["fases"].get(f) or {}).get("hashes") or {})
    actuales = {k: sha256_file(Path(run) / k) for k in H}
    actuales["decisions.md"] = sha256_file(Path(run) / "decisions.md")
    return actuales


def auditar(run, alcance: str | None = None) -> dict:
    from contract import cargar_contrato, cargar_dev, contexto_eval, estado_feature
    from cv import make_splits, validar_splits
    from experiment_log import bitacora, contexto_cv, correr_cv, ajustar_predecir, registros_vigentes, spec_desde_registro
    from gate import estado_fase
    from leakage_check import coincide, desfase_exacto, desplazar, perfil_temporal
    from metrics import METRICAS, calcular
    run = Path(run)
    c = cargar_contrato(run)
    dev = cargar_dev(run)
    st = load_state(run)
    tol = c.get("auditoria") or {}
    rtol, atol = float(tol.get("tolerancia_rel", 1e-6)), float(tol.get("tolerancia_abs", 1e-9))
    alcance = alcance or tol.get("alcance_reejecucion", "seleccionado")
    cmpj = read_json(run / "comparison.json")
    sel = cmpj["congelado"]
    pm = c["metrica"]["principal"]
    K = contexto_cv(run)
    regs = registros_vigentes(run)
    rec_sel = [r for r in regs if r["run_id"] == sel["run_id"]][-1]
    C = []

    # ---- integridad
    malas = [f"{f}: {m}" for f in FASES[: FASES.index("revision")] for e, m in [estado_fase(run, st, f)] if e != "valida"]
    for r in regs:
        if sha256_file(run / r["preds_ref"]) != r["preds_sha256"]:
            malas.append(f"predicciones de {r['run_id']} modificadas")
        if r["tipo"] == "experimento" and sha256_text((run / r["codigo_ref"]).read_text(encoding="utf-8")) != r["codigo_sha256"]:
            malas.append(f"código congelado de {r['run_id']} modificado")
    C.append(_chk("A-INTEGRIDAD", "integridad de artefactos", "falla" if malas else "verificado",
                  "; ".join(malas) if malas else f"{len(regs)} ejecuciones vigentes con predicciones y código íntegros; fases previas válidas",
                  "bloqueante", "Restaurar los artefactos o rehacer las fases afectadas.", "state.json / preds/"))

    # ---- splits
    sp = read_json(run / "splits.json")
    sp2 = make_splits(dev, c)
    probs = validar_splits(dev, sp, c)
    if sp2["hash"] != sp["hash"]:
        probs.append("los splits no se reconstruyen idénticos desde el contrato y los datos")
    man = read_json(run / "holdout_manifest.json")
    ids_h = set(man["ids_holdout"])
    if ids_h & set(dev[c["datos"]["id_fila"]].astype(str)):
        probs.append("hay filas del holdout en desarrollo")
    est = man["estrategia"]
    if est.get("entidades_holdout") and set(est["entidades_holdout"]) & set(dev.get("_entidad", pd.Series(dtype=str))):
        probs.append("hay entidades del holdout en desarrollo")
    if man.get("periodos_holdout") and "_t" in dev:
        gap = int(c["validacion"].get("gap", 0) or 0)
        if int(dev["_t"].max()) > man["periodos_holdout"][0] - gap - 1:
            probs.append("el desarrollo invade el gap previo al holdout")
    C.append(_chk("A-SPLITS", "splits y separación del holdout", "falla" if probs else "verificado",
                  "; ".join(probs) if probs else f"esquema {sp['esquema']} reconstruido idéntico ({len(sp['folds'])} folds); holdout disjunto",
                  "bloqueante", "Re-validar el contrato y rehacer las fases dependientes.", "splits.json"))

    # ---- métricas desde predicciones
    dif = []
    for r in regs:
        p = read_table(run / r["preds_ref"])
        cl = r.get("clases")
        for f in r["metricas_por_fold"]:
            g = p[(p["fold"] == f["fold"]) & (p["semilla"] == f["semilla"])]
            P = g[[f"p__{k}" for k in cl]].values if cl and f"p__{cl[0]}" in g else None
            for k in K["metricas"]:
                v = calcular(k, g["y_true"].values, g["y_pred"].values, P, cl)
                if not np.isclose(v, f[k], rtol=rtol, atol=atol, equal_nan=True):
                    dif.append(f"{r['run_id']} {f['fold']} s{f['semilla']} {k}: registrado {f[k]:.6g} vs recalculado {v:.6g}")
    C.append(_chk("A-METRICAS", "métricas reproducidas desde predicciones", "falla" if dif else "verificado",
                  "; ".join(dif[:5]) + (f" (+{len(dif) - 5})" if len(dif) > 5 else "") if dif else
                  f"{sum(len(r['metricas_por_fold']) for r in regs)} métricas por fold/semilla reproducidas (rtol={rtol}, atol={atol})",
                  "bloqueante", "Investigar la discrepancia: el registro no corresponde a las predicciones.", "experiments.jsonl"))

    # ---- reejecución
    if alcance == "ninguno":
        C.append(_chk("A-REEJECUCION", "re-ejecución del entrenamiento", "no_verificable", "alcance de auditoría = ninguno"))
    else:
        objetivo = regs if alcance == "todos" else [rec_sel]
        dif, n = [], 0
        for r in objetivo:
            try:
                R = correr_cv(K, spec_desde_registro(run, r), r["semillas"])
            except Exception as e:
                dif.append(f"{r['run_id']}: no se pudo re-ejecutar ({e})")
                continue
            orig = {(f["fold"], f["semilla"]): f for f in r["metricas_por_fold"]}
            for f in R["filas"]:
                n += 1
                o = orig.get((f["fold"], f["semilla"]))
                if o is None or not np.isclose(f[pm], o[pm], rtol=max(rtol, 1e-6), atol=atol):
                    dif.append(f"{r['run_id']} {f['fold']} s{f['semilla']}: {pm} registrado {o and o[pm]} vs re-ejecutado {f[pm]}")
        C.append(_chk("A-REEJECUCION", "re-ejecución del entrenamiento", "falla" if dif else "verificado",
                      "; ".join(dif[:5]) if dif else f"{len(objetivo)} candidato(s) re-entrenado(s) desde cero; {n} métricas coinciden",
                      "bloqueante", "El resultado no es reproducible: fijar semillas/aleatoriedad o investigar dependencia del entorno.",
                      "experiments.jsonl"))

    # ---- aislamiento (estático y dinámico)
    sospechas = []
    for r in regs:
        if r["tipo"] == "experimento":
            codigo = (run / r["codigo_ref"]).read_text(encoding="utf-8")
            sin_texto = re.sub(r"#[^\n]*", "", re.sub(r'"""[\s\S]*?"""', "", codigo))   # sin docstrings ni comentarios
            m = PATRON_DATOS.search(sin_texto)
            if m:
                sospechas.append(f"{r['run_id']} contiene `{m.group(0)}` (carga datos o toca holdout por su cuenta)")
    spec_sel = spec_desde_registro(run, rec_sel)
    fo = K["sp"]["folds"][-1] if K["sp"]["esquema"] == "temporal" else K["sp"]["folds"][0]
    tr = dev.iloc[K["pos"][fo["train"]].values]
    va = dev.iloc[K["pos"][fo["val"]].values]
    # prueba del lote: para el elegido y cada experimento, la predicción de una fila no debe cambiar
    # al alterar OTRAS filas del conjunto evaluado
    canarios = [rec_sel] + [r for r in regs if r["tipo"] == "experimento" and r["run_id"] != rec_sel["run_id"]]
    for r in canarios:
        try:
            sp_r = spec_sel if r is rec_sel else spec_desde_registro(run, r)
        except HarnessError as e:
            sospechas.append(f"{r['run_id']}: {e.que}")
            continue
        if not sp_r["features"] or len(va) < 4:
            continue
        r1 = ajustar_predecir(sp_r, c, tr, va, r["semillas"][0], K["clases"], K["umbral"])
        est_ = r1["estimador"]
        mitad = np.arange(len(va)) >= len(va) // 2
        va2 = va.copy()
        rng = np.random.RandomState(1)
        for f in sp_r["features"]:
            # cambiar las estadísticas del lote (no basta permutar: conserva media y desviación)
            vals = va2.loc[va2.index[mitad], f]
            if pd.api.types.is_numeric_dtype(vals):
                va2.loc[va2.index[mitad], f] = vals * 3 + 5 * (float(vals.std()) or 1.0) + 1
            else:
                va2.loc[va2.index[mitad], f] = rng.permutation(vals.values)
        cols = list(getattr(est_, "feature_names_in_", sp_r["features"]))
        if K["clases"]:
            from experiment_log import _proba
            a, b = _proba(est_, va[cols], K["clases"]), _proba(est_, va2[cols], K["clases"])
        else:
            a, b = est_.predict(va[cols]), est_.predict(va2[cols])
        if not np.allclose(np.asarray(a)[~mitad], np.asarray(b)[~mitad], rtol=1e-9, atol=1e-12):
            sospechas.append(f"{r['run_id']}: la predicción de una fila cambia al alterar OTRAS filas del lote "
                             "(alguna transformación se ajusta al predecir: fuga del conjunto evaluado)")
    C.append(_chk("A-AISLAMIENTO", "ajuste dentro de folds", "falla" if sospechas else "verificado",
                  "; ".join(sospechas) if sospechas else "sin carga de datos en el código; predicciones independientes del lote",
                  "bloqueante", "Mover todo lo aprendido dentro del pipeline (fit solo con entrenamiento) y crear un experimento nuevo.",
                  "experiments/_executed/"))

    # ---- disponibilidad / viaje en el tiempo, por feature del elegido
    target = c["datos"]["target"]
    h = int(c["prediccion"].get("horizonte", 0) or 0)
    kmax = min(max(h, int(c["prediccion"].get("estacionalidad") or 0), 7), 28)
    feats_info = c.get("features") or {}
    for f in spec_sel["features"]:
        e, motivo = estado_feature(c, f)
        if e not in ("ok", "excepcion"):
            C.append(_chk(f"A-DISP-{f}", f"disponibilidad de {f}", "falla", motivo, "bloqueante",
                          "Excluir la feature y rehacer la comparación.", "contract.yaml"))
            continue
        if "_t" not in dev:
            C.append(_chk(f"A-DISP-{f}", f"disponibilidad de {f}", "no_verificable",
                          f"declarada {motivo}; sin columna de tiempo no hay forma de reconstruirla al momento de predicción"))
            continue
        k = desfase_exacto(dev, f, target, kmax)
        if k is not None and k < max(h, 1):
            C.append(_chk(f"A-DISP-{f}", f"disponibilidad de {f}", "falla",
                          f"coincide exactamente con {target} desplazado {k} periodos (se requiere >= {max(h, 1)})", "bloqueante",
                          "Es información no disponible al predecir: excluirla.", "contract.yaml"))
            continue
        info = feats_info.get(f) if isinstance(feats_info.get(f), dict) else {}
        lin = info.get("linaje")
        pt = perfil_temporal(dev, f, target, kmax)
        señal = pt and pt["max_futuro"] >= 0.9 and pt["max_futuro"] - pt["max_presente_pasado"] >= 0.1
        if lin and lin.get("fuente") in dev:
            rec_f = desplazar(dev, lin["fuente"], int(lin.get("desfase", 0)))
            if coincide(dev[f], rec_f):
                C.append(_chk(f"A-DISP-{f}", f"disponibilidad de {f}", "verificado",
                              f"reconstruida desde {lin['fuente']} con desfase {lin.get('desfase', 0)} usando solo información previa"))
            else:
                C.append(_chk(f"A-DISP-{f}", f"disponibilidad de {f}", "falla",
                              f"el linaje declarado ({lin['fuente']} desfasada {lin.get('desfase', 0)}) NO reproduce la feature"
                              + (f"; además se asocia con el target {pt['k_futuro']} periodos en el futuro (r={pt['max_futuro']:.3f})" if señal else ""),
                              "bloqueante", "La disponibilidad declarada es falsa o el linaje está mal: investigar el origen y excluir la feature si no se puede demostrar.",
                              "contract.yaml"))
            continue
        if k is not None:
            C.append(_chk(f"A-DISP-{f}", f"disponibilidad de {f}", "verificado", f"coincide con {target} desfasado {k} >= {max(h, 1)} periodos"))
        else:
            C.append(_chk(f"A-DISP-{f}", f"disponibilidad de {f}", "no_verificable",
                          f"declarada {motivo}; sin linaje ni desfase exacto con el target, no se puede reconstruir al momento de predicción"))
            if señal:
                C.append(_chk(f"A-SENAL-{f}", f"asociación temporal de {f}", "falla",
                              f"sin linaje verificable y se asocia más con el target {pt['k_futuro']} periodos en el FUTURO "
                              f"(r={pt['max_futuro']:.3f} vs {pt['max_presente_pasado']:.3f} presente/pasado). Es una señal "
                              "estadística, no una prueba", "importante",
                              "Demostrar con el dueño del dato cuándo se genera; si no se puede, excluirla.", "contract.yaml"))

    # ---- leakage
    L = read_json(run / "leakage_report.json") or {}
    abiertos = [x["id"] for x in L.get("hallazgos", []) if x["estado"] == "abierto"]
    just = [x for x in L.get("hallazgos", []) if x["estado"] == "justificada" and x.get("feature") in spec_sel["features"]]
    if abiertos:
        C.append(_chk("A-LEAKAGE", "hallazgos de leakage", "falla", f"abiertos: {abiertos}", "bloqueante", "Resolverlos.", "leakage_report.json"))
    elif just:
        C.append(_chk("A-LEAKAGE", "hallazgos de leakage", "falla",
                      "el modelo elegido usa features con señales de leakage justificadas por el experimentador: "
                      + "; ".join(f"{x['id']} ({x['justificacion']})" for x in just), "importante",
                      "El revisor debe evaluar si la justificación se sostiene con evidencia.", "leakage_resolutions.yaml"))
    else:
        C.append(_chk("A-LEAKAGE", "hallazgos de leakage", "verificado", f"veredicto {L.get('veredicto')}; el elegido no usa features señaladas"))

    vi = cmpj.get("vs_ingenuo") or {}
    if vi.get("mejora_relativa") is not None and vi["mejora_relativa"] > 0.9:
        C.append(_chk("A-SOSPECHOSO", "resultado sospechosamente bueno", "falla",
                      f"mejora de {vi['mejora_relativa']:.0%} sobre el ingenuo {vi['ingenuo']}", "importante",
                      "Asumir leakage hasta demostrar lo contrario.", "comparison.json"))
    else:
        C.append(_chk("A-SOSPECHOSO", "resultado sospechosamente bueno", "verificado",
                      f"mejora relativa sobre el ingenuo: {vi.get('mejora_relativa')}"))

    ho = st["holdout"]
    ev_h = [e for e in st.get("eventos", []) if "holdout" in e.get("tipo", "")]
    if ho["estado"] not in ("sellado",) or ho.get("congelado") or ev_h:
        C.append(_chk("A-HOLDOUT", "holdout intacto", "falla", f"estado del holdout: {ho['estado']}; eventos: {len(ev_h)}", "bloqueante",
                      "El holdout se abrió antes de terminar la revisión: declarar nueva versión.", "state.json"))
    else:
        C.append(_chk("A-HOLDOUT", "holdout intacto", "verificado", "sellado; ninguna decisión pudo usarlo"
                      + (f"; observaciones previas en otras versiones: {ho['observaciones_previas']}" if ho.get("observaciones_previas") else "")))

    n_exp = len({r["id"] for r in regs if r["tipo"] == "experimento"})
    J = len(K["sp"]["folds"])
    C.append(_chk("A-BUSQUEDA", "intensidad de búsqueda", "falla" if n_exp > 3 * J else "verificado",
                  f"{n_exp} experimentos sobre {J} folds", "importante",
                  "Riesgo de sobreajuste al CV: tratar la estimación CV como optimista y apoyarse en el holdout.", "experiments.jsonl"))
    cierre = [r for r in bitacora(run) if r.get("tipo") == "cierre" and r.get("eval_hash") == contexto_eval(run)["eval_hash"]]
    if cierre:
        C.append(_chk("A-CIERRE", "cierre anticipado de la exploración", "falla",
                      f"motivo {cierre[-1]['motivo']}: {cierre[-1]['evidencia']}", "menor",
                      "El revisor debe evaluar si la evidencia del cierre se sostiene.", "experiments.jsonl"))

    fallas = [x for x in C if x["veredicto"] == "falla"]
    return {"fecha": ahora(), "seleccionado": sel["run_id"], "alcance_reejecucion": alcance, "tolerancias": {"rtol": rtol, "atol": atol},
            "artefactos": conjunto_revisado(run), "verificaciones": C,
            "resumen": {v: sum(1 for x in C if x["veredicto"] == v) for v in ("verificado", "falla", "no_verificable")},
            "bloqueantes": [x["id"] for x in fallas if x["severidad"] == "bloqueante"]}


def vigente(run) -> tuple[bool, str]:
    A = read_json(Path(run) / "audit.json")
    if A is None:
        return False, "no existe audit.json"
    if A["artefactos"] != conjunto_revisado(run):
        return False, "los artefactos cambiaron después de la auditoría"
    return True, ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--alcance", choices=["seleccionado", "todos", "ninguno"])
    a = ap.parse_args()
    from gate import requerir
    requerir(a.run, "revision")
    A = auditar(a.run, a.alcance)
    atomic_write_json(Path(a.run) / "audit.json", A)
    for x in A["verificaciones"]:
        sev = f" [{x['severidad']}]" if x["severidad"] else ""
        print(f"{x['veredicto']:<14}{sev:<14} {x['id']}: {x['evidencia']}")
    print(f"\nResumen: {A['resumen']}")
    if A["bloqueantes"]:
        print(f"BLOQUEADO: fallas bloqueantes {A['bloqueantes']}. Cómo resolverlo: ver `accion` en audit.json.")
        return 1
    return 0


if __name__ == "__main__":
    cli(main)
