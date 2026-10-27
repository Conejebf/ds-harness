"""Fase 0: contrato, separación del holdout, splits y gates de estado."""
import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from conftest import correr
from synth import crear_run


def editar_contrato(run, fn):
    p = Path(run) / "contract.yaml"
    c = yaml.safe_load(p.read_text(encoding="utf-8"))
    fn(c)
    p.write_text(yaml.safe_dump(c, allow_unicode=True, sort_keys=False), encoding="utf-8")


def test_init_deja_contrato_incompleto_que_bloquea(tmp_path):
    run = tmp_path / "r"
    correr("contract.py", "init", "--type", "regression", run=run, check=True)
    r = correr("contract.py", "validate", run=run, check=False)
    assert "BLOQUEADO" in r.stderr and "Cómo resolverlo" in r.stderr
    assert "datos.target" in r.stderr and "comparacion.margen_practico" in r.stderr


def test_contrato_completo_materializa_y_aprueba(tmp_path):
    run = crear_run(tmp_path / "f", "forecasting")
    r = correr("contract.py", "validate", run=run, check=True)
    assert "duplicados eliminados=10" in r.stdout
    st = json.loads((run / "state.json").read_text(encoding="utf-8"))
    assert st["fases"]["contrato"]["estado"] == "aprobada"
    assert st["holdout"]["estado"] == "sellado"
    man = json.loads((run / "holdout_manifest.json").read_text(encoding="utf-8"))
    from contract import cargar_dev
    dev = cargar_dev(run)
    assert set(man["ids_holdout"]).isdisjoint(set(dev["row_id"]))
    assert dev["_t"].max() < man["periodos_holdout"][0]
    assert (run / "_sealed").exists()


@pytest.mark.parametrize("cambio,texto", [
    (lambda c: c["validacion"].update(esquema="kfold"), "dependencia temporal"),
    (lambda c: c["holdout"].update(estrategia="aleatorio"), "holdout no temporal"),
    (lambda c: c["validacion"].update(tamano_validacion=10), "tamano_validacion"),
    (lambda c: c["metrica"].update(principal="roc_auc"), "no es compatible"),
    (lambda c: c["features"]["temp"].update(disponibilidad="desconocida"), "sin excepción documentada"),
    (lambda c: c["features"].pop("promo"), "promo"),
    (lambda c: c["comparacion"].update(margen_practico="POR_DEFINIR"), "margen_practico"),
])
def test_contrato_incoherente_bloquea(tmp_path, cambio, texto):
    run = crear_run(tmp_path / "f", "forecasting")
    editar_contrato(run, cambio)
    r = correr("contract.py", "validate", run=run, check=False)
    assert texto in r.stderr, r.stderr


def test_entidades_nuevas_exige_grupos(tmp_path):
    run = crear_run(tmp_path / "c", "classification")
    editar_contrato(run, lambda c: c["validacion"].update(esquema="estratificado"))
    r = correr("contract.py", "validate", run=run, check=False)
    assert "entidades nuevas" in r.stderr


def test_excepcion_documentada_permite_desconocida(tmp_path):
    run = crear_run(tmp_path / "f", "forecasting")
    editar_contrato(run, lambda c: (c["features"]["temp"].update(disponibilidad="desconocida"),
                                    c["excepciones_disponibilidad"].update(
                                        temp="El proveedor del clima no documenta la hora de publicación; se acepta el riesgo.")))
    correr("contract.py", "validate", run=run, check=True)


def test_estado_feature_disponibilidad():
    from contract import estado_feature
    c = {"prediccion": {"horizonte": 7}, "datos": {"target": "y"},
         "features": {"a": {"disponibilidad": "con_retraso", "retraso": 3},
                      "b": {"disponibilidad": "con_retraso", "retraso": 7},
                      "c": {"disponibilidad": "posterior"},
                      "d": {"disponibilidad": "al_predecir", "linaje": {"fuente": "y", "desfase": 0}},
                      "e": {"disponibilidad": "conocida_anticipada"}}}
    assert estado_feature(c, "a")[0] == "violacion"
    assert estado_feature(c, "b")[0] == "ok"
    assert estado_feature(c, "c")[0] == "violacion"
    assert estado_feature(c, "d")[0] == "violacion"
    assert estado_feature(c, "e")[0] == "ok"


def test_splits_por_tipo(tmp_path):
    from contract import cargar_contrato, cargar_dev
    from cv import validar_splits
    for tipo in ("forecasting", "classification", "regression"):
        run = crear_run(tmp_path / tipo, tipo)
        correr("contract.py", "validate", run=run, check=True)
        sp = json.loads((run / "splits.json").read_text(encoding="utf-8"))
        assert validar_splits(cargar_dev(run), sp, cargar_contrato(run)) == []
        if tipo == "regression":
            assert len(sp["folds"]) == 10
        if tipo == "forecasting":
            assert all(f["origen"] is not None for f in sp["folds"])


def test_validar_splits_detecta_violaciones(tmp_path):
    from common import sha256_json
    from contract import cargar_contrato, cargar_dev
    from cv import validar_splits
    run = crear_run(tmp_path / "c", "classification")
    correr("contract.py", "validate", run=run, check=True)
    sp = json.loads((run / "splits.json").read_text(encoding="utf-8"))
    f0 = sp["folds"][0]
    f0["train"].append(f0["val"][0])   # misma fila (y entidad) en train y val
    sp["hash"] = sha256_json(sp["folds"])
    probs = validar_splits(cargar_dev(run), sp, cargar_contrato(run))
    assert any("a la vez" in p for p in probs) and any("entidades" in p for p in probs)


def test_temporal_gap_violado(tmp_path):
    from contract import cargar_contrato, cargar_dev
    from cv import validar_splits
    run = crear_run(tmp_path / "f", "forecasting", validacion__gap=2, prediccion__retraso_target=2)
    correr("contract.py", "validate", run=run, check=True)
    c = cargar_contrato(run)
    dev = cargar_dev(run)
    sp = json.loads((run / "splits.json").read_text(encoding="utf-8"))
    assert validar_splits(dev, sp, c) == []
    c["validacion"]["gap"] = 5   # contrato más estricto que los splits materializados
    assert any("gap" in p for p in validar_splits(dev, sp, c))


def test_gate_bloquea_fases_sin_previas_y_detecta_cambios(tmp_path):
    run = crear_run(tmp_path / "f", "forecasting")
    r = correr("gate.py", "check", "eda", run=run, check=False)
    assert "contrato" in r.stderr and "Cómo resolverlo" in r.stderr
    correr("contract.py", "validate", run=run, check=True)
    correr("gate.py", "check", "eda", run=run, check=True)
    r = correr("gate.py", "check", "baselines", run=run, check=False)
    assert "'eda'" in r.stderr
    p = run / "data" / "ventas.csv"
    df = pd.read_csv(p)
    df.loc[0, "ventas"] += 1
    df.to_csv(p, index=False)
    r = correr("gate.py", "check", "eda", run=run, check=False)
    assert "data/ventas.csv cambió" in r.stderr
    assert "invalida" in correr("gate.py", "status", run=run, check=True).stdout


def test_editar_hash_en_state_se_detecta(tmp_path):
    run = crear_run(tmp_path / "f", "forecasting")
    correr("contract.py", "validate", run=run, check=True)
    st = json.loads((run / "state.json").read_text(encoding="utf-8"))
    st["fases"]["contrato"]["hashes"]["splits.json"] = "0" * 64
    (run / "state.json").write_text(json.dumps(st), encoding="utf-8")
    correr("gate.py", "check", "eda", run=run, check=False)


def test_lock_concurrente(tmp_path):
    from common import HarnessError, file_lock
    with file_lock(tmp_path):
        with pytest.raises(HarnessError):
            with file_lock(tmp_path, timeout=0.3):
                pass
