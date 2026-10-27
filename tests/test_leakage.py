"""Fase 1: perfil y leakage. Fugas plantadas se detectan; casos limpios no se bloquean."""
import json
from pathlib import Path

import pandas as pd
import pytest

from conftest import correr
from helpers import editar_contrato, escribir_interpretacion, excluir
from synth import crear_run


def _leak(run):
    return json.loads((Path(run) / "leakage_report.json").read_text(encoding="utf-8"))


def _ids(run):
    return {x["id"]: x for x in _leak(run)["hallazgos"]}


def _preparar(tmp_path, tipo, con_fuga=True, **aj):
    run = crear_run(tmp_path / tipo, tipo, con_fuga=con_fuga, **aj)
    correr("contract.py", "validate", run=run, check=True)
    correr("profile_data.py", run=run, check=True)
    return run


def test_fugas_plantadas_forecasting_se_detectan_y_bloquean(tmp_path):
    run = _preparar(tmp_path, "forecasting")
    r = correr("leakage_check.py", run=run, check=False)
    assert "Veredicto de leakage: bloqueado" in r.stdout
    ids = _ids(run)
    assert "S-UNIVAR-ventas_ajustadas" in ids          # derivada del target
    assert "S-TEMPORAL-indice_demanda" in ids          # información futura disfrazada
    assert not any("ventas_lag" in i or "media_movil" in i for i in ids)  # rezagos legítimos no se bloquean
    assert "Veredicto de leakage: bloqueado" in (run / "eda_report.md").read_text(encoding="utf-8")
    escribir_interpretacion(run)
    r = correr("gate.py", "advance", "eda", run=run, check=False)
    assert "leakage bloqueado" in r.stderr and "Cómo resolverlo" in r.stderr


@pytest.mark.parametrize("tipo,feature", [("classification", "motivo_baja_registrado"), ("regression", "y_copia_ruidosa")])
def test_fugas_plantadas_tabulares(tmp_path, tipo, feature):
    run = _preparar(tmp_path, tipo)
    correr("leakage_check.py", run=run, check=False)
    assert any(x["feature"] == feature and x["estado"] == "abierto" for x in _leak(run)["hallazgos"])


@pytest.mark.parametrize("tipo", ["forecasting", "classification", "regression"])
def test_datos_limpios_no_se_bloquean(tmp_path, tipo):
    run = _preparar(tmp_path, tipo, con_fuga=False)
    r = correr("leakage_check.py", run=run, check=True)
    assert "limpio" in r.stdout
    escribir_interpretacion(run)
    correr("gate.py", "advance", "eda", run=run, check=True)


def test_futuro_exacto_es_violacion_comprobada(tmp_path):
    run = crear_run(tmp_path / "f", "forecasting", con_fuga=False)
    p = run / "data" / "ventas.csv"
    df = pd.read_csv(p)
    df["ventas_manana"] = df.groupby("tienda")["ventas"].shift(-1)
    df.to_csv(p, index=False)
    editar_contrato(run, lambda c: c["features"].update(ventas_manana={"disponibilidad": "al_predecir"}))
    correr("contract.py", "validate", run=run, check=True)
    correr("profile_data.py", run=run, check=True)
    correr("leakage_check.py", run=run, check=False)
    h = _ids(run)["V-DESFASE-ventas_manana"]
    assert h["clase"] == "violacion" and "FUTURO" in h["evidencia"]
    # justificar una violación no la cierra
    (run / "leakage_resolutions.yaml").write_text(
        "V-DESFASE-ventas_manana:\n  estado: justificada\n  justificacion: 'confío en que el dato está disponible a tiempo'\n",
        encoding="utf-8")
    correr("leakage_check.py", run=run, check=False)
    R = _leak(run)
    assert R["veredicto"] == "bloqueado" and any("no se justifican" in e for e in R["errores_resoluciones"])


def test_retraso_insuficiente_declarado_es_violacion(tmp_path):
    run = crear_run(tmp_path / "f", "forecasting", con_fuga=False)
    editar_contrato(run, lambda c: c["features"]["ventas_lag7"].update(retraso=3))
    correr("contract.py", "validate", run=run, check=True)
    correr("profile_data.py", run=run, check=True)
    correr("leakage_check.py", run=run, check=False)
    assert _ids(run)["V-DISP-ventas_lag7"]["clase"] == "violacion"


def test_senal_justificada_permite_avanzar_con_excepciones(tmp_path):
    run = _preparar(tmp_path, "regression")
    (run / "leakage_resolutions.yaml").write_text(
        "\n".join(f"{i}:\n  estado: justificada\n  justificacion: 'medición independiente tomada antes del evento; validado con el dueño del dato'"
                  for i in ("S-UNIVAR-y_copia_ruidosa", "S-CORR-y_copia_ruidosa", "S-IMPORTANCIA-y_copia_ruidosa")),
        encoding="utf-8")
    r = correr("leakage_check.py", run=run, check=True)
    assert "aprobado_con_excepciones" in r.stdout


def test_exclusion_resuelve_y_gate_exige_interpretacion(tmp_path):
    run = _preparar(tmp_path, "forecasting")
    correr("leakage_check.py", run=run, check=False)
    excluir(run, ["ventas_ajustadas", "indice_demanda"])
    correr("contract.py", "validate", run=run, check=True)
    correr("profile_data.py", run=run, check=True)
    r = correr("leakage_check.py", run=run, check=True)
    assert "limpio" in r.stdout
    r = correr("gate.py", "advance", "eda", run=run, check=False)
    assert "Interpretación" in r.stderr
    escribir_interpretacion(run)
    correr("gate.py", "advance", "eda", run=run, check=True)
    # cambiar el contrato después invalida contrato y EDA
    editar_contrato(run, lambda c: c["comparacion"].update(alfa=0.1))
    assert "cambió" in correr("gate.py", "check", "baselines", run=run, check=False).stderr


def test_duplicados_legitimos_que_cruzan_folds_son_senal(tmp_path):
    run = crear_run(tmp_path / "r", "regression", con_fuga=False, datos__duplicados="legitimos")
    p = run / "data" / "tabla.csv"
    df = pd.read_csv(p)
    d = df.sample(30, random_state=0).copy()
    d["row_id"] = [f"X{i}" for i in range(30)]
    pd.concat([df, d]).to_csv(p, index=False)
    correr("contract.py", "validate", run=run, check=True)
    correr("profile_data.py", run=run, check=True)
    correr("leakage_check.py", run=run, check=False)
    assert "S-DUP" in _ids(run)


def test_perfil_exploratorio_con_contrato_incompleto(tmp_path):
    run = tmp_path / "x"
    correr("contract.py", "init", "--type", "regression", run=run, check=True)
    from synth import regression
    regression().to_csv(run / "data" / "t.csv", index=False)
    r = correr("profile_data.py", "--exploratorio", "--datos", run / "data" / "t.csv", run=run, check=True)
    assert "no aprueba" in r.stdout
    correr("profile_data.py", run=run, check=False)   # el perfil normal sí exige contrato
