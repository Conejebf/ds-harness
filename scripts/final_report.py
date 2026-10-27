"""
Fase 7 — Reporte final (Markdown en español + figuras), generado SOLO desde artefactos aprobados.

Uso:
    python scripts/final_report.py --run runs/<p>

Se niega a generarse (exit != 0, diciendo qué falta y cómo resolverlo) si cualquier fase previa no
está aprobada y vigente, si hay bloqueantes abiertos en la revisión o si el holdout no se evaluó.

Contenido: resumen ejecutivo; objetivo y escenario de predicción; datos (periodo, procedencia,
limitaciones); splits y controles de leakage; escalera de baselines, hipótesis y experimentos
(incluidos los fallidos y descartados); comparación con incertidumbre y criterio de elección;
diagnósticos; revisión adversarial; resultado del holdout SEPARADO del CV; recomendación operativa
y condiciones de uso; riesgos de drift y monitoreo; limitaciones; cómo reproducir.
No oculta resultados negativos ni presenta mejoras de CV como rendimiento garantizado.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import HarnessError, atomic_write_text, cli, load_state, read_json
from experiment_log import bitacora


def _f(v, d=4):
    if v is None:
        return "—"
    if isinstance(v, float) and not np.isfinite(v):
        return "—"
    return f"{v:.{d}g}" if isinstance(v, (int, float, np.floating)) else str(v)


def _figura_escalera(cmpj: dict, dest: Path) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    C = sorted([x for x in cmpj["candidatos"] if x["elegible"]], key=lambda x: x["valor"], reverse=cmpj["mayor_mejor"])
    dest.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 0.35 * len(C) + 1.2))
    y = np.arange(len(C))
    con_ic = all(x.get("ic_metrica") for x in C)
    if con_ic:
        err = [[x["valor"] - x["ic_metrica"][0] for x in C], [x["ic_metrica"][1] - x["valor"] for x in C]]
    else:
        err = [x["std_entre_folds"] or 0 for x in C]
    plt.errorbar([x["valor"] for x in C], y, xerr=err, fmt="o", capsize=3)
    plt.yticks(y, [f"{x['run_id']} ({x['familia']})" for x in C])
    sel = cmpj["seleccion"]["run_id"]
    for i, x in enumerate(C):
        if x["run_id"] == sel:
            plt.plot(x["valor"], i, "r*", ms=14)
    plt.xlabel(f"{cmpj['metrica']} " + ("(IC de la métrica)" if con_ic else "(media ± desv. entre folds; sin IC válido)"))
    plt.title("Escalera de modelos (★ = elegido)")
    plt.tight_layout()
    plt.savefig(dest / "escalera.png", dpi=90)
    plt.close()
    return "final_report_figuras/escalera.png"


def generar(run) -> str:
    from contract import cargar_contrato
    from gate import requerir
    run = Path(run)
    requerir(run, "reporte")
    R = read_json(run / "review.json")
    ab = [k for k, v in R["hallazgos"].items() if v["estado"] == "abierto" and v["severidad"] == "bloqueante"]
    if ab:
        raise HarnessError(f"hay bloqueantes abiertos en la revisión: {ab}.", "Resuélvelos y vuelve a revisar.")
    st = load_state(run)
    c = cargar_contrato(run)
    snap, man = read_json(run / "snapshot.json"), read_json(run / "holdout_manifest.json")
    prof, leak = read_json(run / "eda" / "profile.json"), read_json(run / "leakage_report.json")
    B, cmpj = read_json(run / "baselines.json"), read_json(run / "comparison.json")
    D, A, H = read_json(run / "diagnostics" / "diagnostics.json"), read_json(run / "audit.json"), read_json(run / "holdout_result.json")
    sp = read_json(run / "splits.json")
    pm = c["metrica"]["principal"]
    sel = cmpj["seleccion"]
    cand = next(x for x in cmpj["candidatos"] if x["run_id"] == sel["run_id"])
    regs = bitacora(run)
    concl = {r["id"]: r for r in regs if r.get("tipo") == "conclusion"}
    exps = [r for r in regs if r.get("tipo") == "experimento"]
    fig = _figura_escalera(cmpj, run / "final_report_figuras")
    ref = H.get("referencia_ingenuo")
    mayor = cmpj["mayor_mejor"]
    hval = H["metricas"][pm]["valor"]

    # ---------------- recomendación operativa (derivada de hechos, no de opinión)
    rec, condiciones = [], []
    supera_ing_holdout = ref is None or ref["p_no_mejora"] < float(c["comparacion"].get("alfa", 0.05))
    if not supera_ing_holdout:
        rec.append(f"**No desplegar todavía**: en el holdout el modelo no supera con evidencia al ingenuo {ref['run_id']} "
                   f"(mejora {_f(ref['mejora'])}, IC [{_f(ref['ic_mejora'][0])}, {_f(ref['ic_mejora'][1])}]).")
    elif sel["estado"] in ("provisional", "operativa"):
        rec.append(f"Usar {sel['run_id']} con cautela: la elección es **{sel['estado']}** ({sel['criterio']}).")
    else:
        rec.append(f"Usar {sel['run_id']}: la elección es concluyente en CV y el holdout la respalda frente al ingenuo.")
    if H["observaciones_previas"]:
        rec.append("El holdout ya había sido observado en una versión anterior: su resultado NO es una estimación limpia; conseguir datos nuevos antes de comprometer metas.")
    condiciones.append(f"Mismo escenario del contrato: {c['prediccion']['generalizacion']}, momento de predicción «{c['prediccion']['momento']}»"
                       + (f", horizonte {c['prediccion']['horizonte']} periodos." if c["prediccion"].get("horizonte") else "."))
    condiciones.append(f"Las features deben estar disponibles como se declaró en el contrato: {', '.join(cmpj['congelado']['features']) or 'ninguna'}.")
    nv = [x["id"].replace("A-DISP-", "") for x in A["verificaciones"] if x["id"].startswith("A-DISP-") and x["veredicto"] == "no_verificable"]
    if nv:
        condiciones.append(f"Confirmar con los dueños del dato la disponibilidad de {nv} (la auditoría no pudo reconstruirlas al momento de predicción).")
    drift = (prof.get("drift_mitades") or {}).get("marcadas") or []

    L = []
    L += [f"# Reporte final — {c['estudio']}", "",
          f"Versión del estudio: {st.get('version_estudio')} · Generado: {H['fecha'][:10]} · Tipo: {c['problem_type']}", ""]
    L += ["## 1. Resumen ejecutivo", "",
          f"- Modelo elegido: **{sel['run_id']}** ({'' if cand['nombre'] == cand['id'] else cand['nombre'] + ', '}familia {cand['familia']}, {cand['n_features']} features). "
          f"Estado de la elección: **{sel['estado']}**.",
          f"- CV ({sp['esquema']}, {len(sp['folds'])} folds): {pm} = {_f(cand['valor'])} "
          + (f"IC [{_f(cand['ic_metrica'][0])}, {_f(cand['ic_metrica'][1])}] ({cmpj['metodo']})." if cand.get("ic_metrica")
             else f"(desv. entre folds {_f(cand['std_entre_folds'])}; sin IC válido: {cmpj['motivo_sin_inferencia']})."),
          f"- Holdout (evaluado una sola vez, {H['n_filas']} filas): {pm} = **{_f(hval)}** IC{int(100 - 100 * float(c['comparacion'].get('alfa', 0.05)))} "
          f"[{_f(H['ic_principal'][0])}, {_f(H['ic_principal'][1])}]; diferencia vs CV {_f(H['diferencia_vs_cv'])}.",
          (f"- Frente al ingenuo {ref['run_id']} en holdout: {pm} {_f(ref['valor'])}; mejora {_f(ref['mejora'])} "
           f"IC [{_f(ref['ic_mejora'][0])}, {_f(ref['ic_mejora'][1])}]." if ref else "- El elegido es el propio ingenuo."),
          f"- Recomendación: {' '.join(rec)}", ""]
    L += ["## 2. Objetivo y escenario de predicción", "",
          f"- Target: `{c['datos']['target']}`; unidad: {c['datos']['unidad_observacion']}; entidad: {c['datos'].get('entidad')}; tiempo: {c['datos'].get('tiempo')}.",
          f"- Momento de predicción: {c['prediccion']['momento']}; horizonte: {c['prediccion'].get('horizonte')}; retraso del target: {c['prediccion'].get('retraso_target')}.",
          f"- Generalización: {c['prediccion']['generalizacion']}.",
          f"- Métrica principal: {pm} ({'mayor' if mayor else 'menor'} es mejor), agregación {cmpj['agregacion']}; secundarias: {c['metrica'].get('secundarias')}.",
          f"- Margen práctico para preferir un modelo más simple: {c['comparacion']['margen_practico']} (unidades de {pm}).",
          f"- Restricciones: {c.get('restricciones')}.", ""]
    L += ["## 3. Datos", "",
          f"- Snapshot `{snap['ruta']}` (sha256 {snap['sha256'][:12]}…): {snap['filas']} filas, {snap['columnas']} columnas. Procedencia: {snap['procedencia']}.",
          f"- Desarrollo: {man['filas_dev']} filas; holdout sellado: {man['filas_holdout']} filas ({man['estrategia']['estrategia']}).",
          f"- Duplicados exactos en el snapshot: {man['duplicados']['duplicados_exactos']} (política: {c['datos']['duplicados']}; eliminados: {len(man['duplicados']['ids_duplicados_eliminados'])}).",
          f"- Faltantes: {prof.get('faltantes') or 'ninguno'}. Outliers (3xIQR): {prof.get('outliers_3iqr') or 'ninguno'}.",
          (f"- Periodo: {prof['tiempo']['min']} a {prof['tiempo']['max']} ({prof['tiempo']['periodos']} periodos en desarrollo)." if prof.get("tiempo") else ""),
          f"- Columnas con drift entre mitades del periodo: {drift or 'ninguna'}.",
          f"- Columnas excluidas: {c['datos'].get('columnas_excluidas') or 'ninguna'}.", ""]
    L += ["## 4. Validación y controles de leakage", "",
          f"- Esquema: {sp['esquema']} — {c['validacion']['justificacion']}",
          f"- Folds: {len(sp['folds'])}" + (f"; ventanas de {c['validacion']['tamano_validacion']} periodos con gap {c['validacion'].get('gap', 0)}." if sp["esquema"] == "temporal" else "."),
          f"- Veredicto de leakage: **{leak['veredicto']}**. Hallazgos:"]
    L += [f"  - {x['id']} ({x['clase']}, {x['estado']}): {x['evidencia']}" + (f" — justificación: {x['justificacion']}" if x.get("justificacion") else "")
          for x in leak["hallazgos"]] or ["  - ninguno"]
    L += [f"- Auditoría determinista: {A['resumen']}.", ""]
    L += ["## 5. Baselines, hipótesis y experimentos", "", f"![escalera]({fig})", "",
          "| candidato | tipo | familia | #feat | " + pm + " | IC " + pm + " | desv. folds | desv. semillas | degradación vs mejor | IC degradación | veredicto |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for x in sorted(cmpj["candidatos"], key=lambda x: x["deg"]):
        ic = x.get("ic")
        im = x.get("ic_metrica")
        L.append(f"| {x['run_id']} {x['nombre']} | {x['tipo']} | {x['familia']} | {x['n_features']} | {_f(x['valor'])} | "
                 f"{'—' if not im else f'[{_f(im[0])}, {_f(im[1])}]'} | {_f(x['std_entre_folds'])} | "
                 f"{_f(x['std_entre_semillas'])} | {_f(x['deg'])} | {'—' if not ic else f'[{_f(ic[0])}, {_f(ic[1])}]'} | "
                 f"{x.get('veredicto', '—') if x['elegible'] else 'no elegible: ' + str(x['motivo_no_elegible'])} |")
    L += ["", "Hipótesis (incluidas las descartadas y fallidas):", ""]
    for r in exps:
        h = r["hipotesis"]
        cc = concl.get(r["id"], {})
        L.append(f"- **{r['run_id']}** [{h['categoria']}] {h['hipotesis']} — cambio: {h['cambio']}; esperado: {h['esperado']}. "
                 + (f"Estado: FALLIDO ({r['error']})." if r["estado"] != "ok" else
                    f"Observado: {cc.get('resultado_observado', 'sin registrar')} → **{cc.get('decision', '—')}**."))
    cierre = [r for r in regs if r.get("tipo") == "cierre"]
    if cierre:
        L.append(f"- Cierre anticipado de la exploración: {cierre[-1]['motivo']} — {cierre[-1]['evidencia']}")
    L.append("")
    L += ["## 6. Comparación, incertidumbre y criterio de elección", "",
          f"- Método: {cmpj['metodo']} {cmpj['detalle_metodo']}; inferencia válida: {cmpj['inferencia_valida']}"
          + (f" ({cmpj['motivo_sin_inferencia']})" if cmpj["motivo_sin_inferencia"] else "") + ".",
          f"- Mejor estimado: {cmpj['mejor']}. Elegido: {sel['run_id']} — {sel['criterio']}.",
          f"- Contra el ingenuo (CV): {cmpj.get('vs_ingenuo')}.", f"- {cmpj['nota_ic']}"]
    L += [f"- AVISO: {a}" for a in cmpj["advertencias"]]
    L.append("")
    L += ["## 7. Diagnósticos", "", "| id | resultado | evidencia |", "|---|---|---|"]
    L += [f"| {x['id']} | {x['resultado']} | {x['evidencia']}" + (f" ({x['justificacion']})" if x.get("justificacion") else "") + " |" for x in D["resultados"]]
    L.append("")
    u = R["rondas"][-1]
    L += ["## 8. Revisión adversarial", "",
          f"- Rondas: {len(R['rondas'])}; última por {u['revisor']} ({u['modelo']}): {u['resumen']}"]
    L += [f"- {k} [{v['severidad']}, {v['estado']}, origen {v['origen']}]: {v['evidencia']}" + (f" — resolución: {v['resolucion']}" if v.get("resolucion") else "")
          for k, v in R["hallazgos"].items()] or ["- sin hallazgos"]
    L.append("")
    L += ["## 9. Resultado del holdout (separado del CV)", "",
          f"- Procedimiento: {H['procedimiento']}; ventanas: {len(H['ventanas'])}; unidades de bootstrap: {H['unidades_bootstrap']}.",
          f"- {pm}: {_f(hval)} (por semilla {H['metricas'][pm]['por_semilla']}); IC [{_f(H['ic_principal'][0])}, {_f(H['ic_principal'][1])}].",
          f"- Métricas secundarias: " + ", ".join(f"{k}={_f(v['valor'])}" for k, v in H["metricas"].items() if k != pm) + "."]
    if H["observaciones_previas"]:
        L.append(f"- ATENCIÓN: este holdout ya fue observado antes ({H['observaciones_previas']}); el resultado está contaminado por esa observación.")
    L.append("")
    L += ["## 10. Recomendación operativa y condiciones de uso", ""] + [f"- {x}" for x in rec + condiciones] + [""]
    L += ["## 11. Riesgos de drift y monitoreo", "",
          f"- Columnas con drift ya visible en desarrollo: {drift or 'ninguna'}.",
          f"- Monitorear {pm} semanal/periódicamente contra el valor de holdout ({_f(hval)}) y alertar si sale del IC [{_f(H['ic_principal'][0])}, {_f(H['ic_principal'][1])}].",
          f"- Monitorear la distribución de las features usadas ({', '.join(cmpj['congelado']['features']) or 'n/a'}) con PSI/KS contra desarrollo.",
          "- Mantener el ingenuo de referencia corriendo en paralelo: si el modelo deja de superarlo, reentrenar o revisar.", ""]
    L += ["## 12. Limitaciones", "",
          "- La estimación de CV del elegido es optimista por selección entre varios candidatos; la estimación confirmatoria es el holdout.",
          "- Los IC reflejan la variabilidad de muestreo de las unidades remuestreadas, no cambios futuros del proceso (drift).",
          f"- Features cuya disponibilidad no se pudo verificar: {nv or 'ninguna'}.",
          "- Las señales estadísticas de leakage no prueban ni descartan fugas; ver leakage_report.json y audit.json.",
          "- Los controles de integridad del harness detectan cambios accidentales; no impiden manipulación deliberada.", ""]
    ent = (A.get("artefactos") or {})
    L += ["## 13. Cómo reproducir", "", "```",
          f"python scripts/contract.py validate --run {run}",
          f"python scripts/baselines.py --run {run}",
          *[f"python scripts/experiment_log.py run {r['id']} --run {run}" for r in exps if r["estado"] == "ok" and not r.get("reproduccion_de")],
          f"python scripts/compare_models.py --run {run}",
          f"python scripts/audit.py --run {run} --alcance todos", "```",
          f"Entorno del candidato: {next(r for r in regs if r.get('run_id') == sel['run_id'])['entorno']}.",
          "Tolerancias de reproducción: " + str(A["tolerancias"]) + " (no se promete igualdad bit a bit entre plataformas).", ""]
    return "\n".join(x for x in L if x is not None)


def criterios(run):
    p = Path(run) / "final_report.md"
    if not p.exists():
        return [("falta final_report.md.", f"Corre `python scripts/final_report.py --run {run}`.")], []
    return [], ["final_report.md"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    a = ap.parse_args()
    from gate import registrar
    md = generar(a.run)
    atomic_write_text(Path(a.run) / "final_report.md", md + "\n")
    registrar(a.run, "reporte", ["final_report.md"])
    print(f"OK: reporte final en {Path(a.run) / 'final_report.md'}")
    return 0


if __name__ == "__main__":
    cli(main)
