"""
End-to-end: los tres tipos de problema recorren todas las fases con datos sintéticos que traen
fugas plantadas. Demuestra que (a) los gates bloquean cuando faltan artefactos, (b) leakage_check
detecta las fugas plantadas, y que el ciclo del holdout no permite reutilizarlo en silencio.
(c) — el revisor encuentra la fuga que se escapa — está en test_audit_review.py (determinista)
y en test_llm_reviewer.py (LLM real, opcional).
"""
import json
import shutil
from pathlib import Path

import pytest

from conftest import correr
from experimentos import EXPERIMENTOS
from helpers import (FUGAS, escribir_interpretacion, excluir, experimentar, hasta_diagnostico, leer, ronda_revision)
from synth import crear_run


@pytest.mark.parametrize("tipo", ["forecasting", "classification", "regression"])
def test_recorrido_completo(tmp_path, tipo):
    run = crear_run(tmp_path / tipo, tipo)

    # (a) sin contrato validado nada arranca, y el reporte se niega
    for paso in (("profile_data.py",), ("baselines.py",), ("final_report.py",), ("holdout.py", "evaluate")):
        r = correr(*paso, run=run, check=False)
        assert "BLOQUEADO" in r.stderr and "Cómo resolverlo" in r.stderr, paso

    correr("contract.py", "validate", run=run, check=True)
    correr("profile_data.py", run=run, check=True)

    # (b) la fuga plantada se detecta y bloquea la fase
    r = correr("leakage_check.py", run=run, check=False)
    rep = leer(run, "leakage_report.json")
    detectadas = {x["feature"] for x in rep["hallazgos"] if x["estado"] == "abierto"}
    assert set(FUGAS[tipo]) <= detectadas, rep["hallazgos"]
    escribir_interpretacion(run)
    assert "leakage bloqueado" in correr("gate.py", "advance", "eda", run=run, check=False).stderr
    assert "'eda'" in correr("baselines.py", run=run, check=False).stderr

    # resolución: excluir, re-validar, volver a revisar
    excluir(run, FUGAS[tipo])
    correr("contract.py", "validate", run=run, check=True)
    correr("profile_data.py", run=run, check=True)
    correr("leakage_check.py", run=run, check=True)
    correr("gate.py", "advance", "eda", run=run, check=True)

    correr("baselines.py", run=run, check=True)
    correr("gate.py", "advance", "baselines", run=run, check=True)
    instalar_uno = EXPERIMENTOS[tipo]["E001"]
    (run / "experiments" / "E001.py").write_text(instalar_uno, encoding="utf-8")
    correr("experiment_log.py", "run", "E001", run=run, check=True)
    correr("compare_models.py", run=run, check=True)
    r = correr("gate.py", "advance", "experimentos", run=run, check=False)   # 1 hipótesis y sin conclusión
    assert "hipótesis distintas" in r.stderr and "sin resultado observado" in r.stderr
    correr("experiment_log.py", "conclude", "E001", "--resultado", "Resultado observado con cifras en experiments.jsonl frente a la referencia.",
           "--decision", "descartada", run=run, check=True)
    experimentar(run, tipo, ("E002", "E003", "E004", "E005"))
    assert "BLOQUEADO" in correr("audit.py", run=run, check=False).stderr
    hasta_diagnostico(run)
    assert "'revision'" in correr("holdout.py", "evaluate", run=run, check=False).stderr
    ronda_revision(run, check_submit=True)
    correr("gate.py", "advance", "revision", run=run, check=True)

    # holdout: una sola evaluación
    r1 = correr("holdout.py", "evaluate", run=run, check=True)
    r2 = correr("holdout.py", "evaluate", run=run, check=True)
    assert "no se recalcula" in r2.stdout
    C = leer(run, "comparison.json")
    sel = next(x for x in C["candidatos"] if x["run_id"] == C["seleccion"]["run_id"])
    assert sel["ic_metrica"] is None or sel["ic_metrica"][0] <= sel["valor"] <= sel["ic_metrica"][1]
    H = leer(run, "holdout_result.json")
    assert H["ic_principal"][0] <= H["metricas"][H["metrica_principal"]]["valor"] <= H["ic_principal"][1]

    correr("final_report.py", run=run, check=True)
    txt = (run / "final_report.md").read_text(encoding="utf-8")
    for sec in ("Resumen ejecutivo", "Objetivo y escenario", "Datos", "controles de leakage", "Baselines, hipótesis",
                "incertidumbre", "Diagnósticos", "Revisión adversarial", "holdout (separado del CV)", "Recomendación operativa",
                "drift y monitoreo", "Limitaciones", "Cómo reproducir"):
        assert sec in txt, sec
    for f in FUGAS[tipo]:
        assert f in txt          # las exclusiones quedan documentadas
    st = correr("gate.py", "status", run=run, check=True).stdout
    assert st.count("valida") == 8 and "invalida" not in st

    if tipo == "regression":
        _ciclo_holdout(run)


def _ciclo_holdout(run: Path):
    """El holdout no se reutiliza en silencio para elegir otro modelo."""
    codigo = EXPERIMENTOS["regression"]["E003"].replace("Con más iteraciones y tasa mayor", "Con tasa todavía mayor") \
        .replace("'learning_rate': 0.1", "'learning_rate': 0.2")
    (run / "experiments" / "E006.py").write_text(codigo, encoding="utf-8")
    correr("experiment_log.py", "run", "E006", run=run, check=True)
    correr("compare_models.py", run=run, check=True)
    r = correr("holdout.py", "evaluate", run=run, check=False)
    assert "ya se evaluó con otro candidato" in r.stderr and "--new-version" in r.stderr
    r = correr("contract.py", "validate", run=run, check=False)
    assert "nueva versión" in r.stderr
    correr("contract.py", "validate", "--new-version", run=run, check=True)
    st = json.loads((run / "state.json").read_text(encoding="utf-8"))
    assert st["version_estudio"] == 2 and st["holdout"]["estado"] == "sellado"
    assert st["holdout"]["observaciones_previas"][0]["version"] == 1
    assert (run / "versiones" / "v1" / "holdout_result.json").exists() and not (run / "holdout_result.json").exists()
    assert "pendiente" in correr("gate.py", "status", run=run, check=True).stdout


def test_recuperacion_del_holdout(tmp_path):
    from helpers import hasta_revision
    run = crear_run(tmp_path / "r", "regression")
    hasta_revision(run, "regression", ids=("E001", "E002", "E003", "E004", "E005"))
    # el archivo sellado faltante lo detecta el gate ANTES de abrir el holdout
    sellado = next((run / "_sealed").glob("holdout.*"))
    tmp = sellado.with_name("oculto.bin")
    shutil.move(sellado, tmp)
    assert "'contrato'" in correr("holdout.py", "evaluate", run=run, check=False).stderr
    shutil.move(tmp, sellado)
    assert json.loads((run / "state.json").read_text(encoding="utf-8"))["holdout"]["estado"] == "sellado"
    # falla A MITAD de la evaluación: un directorio ocupa el nombre del archivo de predicciones
    from common import TABLE_EXT
    estorbo = run / f"holdout_preds{TABLE_EXT}"
    estorbo.mkdir()
    r = correr("holdout.py", "evaluate", run=run, check=False)
    assert "en_curso" in r.stderr and "--recuperar" in r.stderr
    estorbo.rmdir()
    r = correr("holdout.py", "evaluate", run=run, check=False)
    assert "interrumpida" in r.stderr
    correr("holdout.py", "evaluate", "--recuperar", run=run, check=True)
    st = json.loads((run / "state.json").read_text(encoding="utf-8"))
    assert [e["tipo"] for e in st["holdout"]["eventos"]] == ["apertura", "error", "recuperacion", "evaluado"]
