"""Fase 4: diagnóstico estructurado y su gate."""
import json
import shutil

import pytest

from conftest import correr
from helpers import DECLARACION, agregar_decision, experimentar, hasta_eda, leer
from synth import crear_run


@pytest.fixture(scope="module")
def base_exp(tmp_path_factory):
    run = crear_run(tmp_path_factory.mktemp("dg") / "reg", "regression")
    hasta_eda(run, "regression")
    correr("baselines.py", run=run, check=True)
    correr("gate.py", "advance", "baselines", run=run, check=True)
    experimentar(run, "regression")
    correr("compare_models.py", run=run, check=True)
    correr("gate.py", "advance", "experimentos", run=run, check=True)
    correr("diagnostics.py", run=run, check=True)
    return run


@pytest.fixture
def run(base_exp, tmp_path):
    dst = tmp_path / "reg"
    shutil.copytree(base_exp, dst)
    return dst


def test_resultados_estructurados(run):
    from diagnostics import REQUERIDOS, RESULTADOS
    D = leer(run, "diagnostics/diagnostics.json")
    ids = [x["id"] for x in D["resultados"]]
    for k in REQUERIDOS["regression"]:
        assert any(i == k or i.startswith(k + "-") for i in ids), k
    assert all(x["resultado"] in RESULTADOS and x["evidencia"] for x in D["resultados"])
    assert all(x["justificacion"] for x in D["resultados"] if x["resultado"] == "no_evaluable")
    assert (run / "diagnostics" / "figuras" / "residuos.png").exists()
    assert (run / "diagnostics" / "diagnostics.md").read_text(encoding="utf-8").count("|") > 20


def test_sin_accionables_exige_declaracion(run):
    p = run / "diagnostics" / "diagnostics.json"
    D = json.loads(p.read_text(encoding="utf-8"))
    for x in D["resultados"]:
        if x["resultado"] == "accionable":
            x["resultado"] = "sin_hallazgos"
    p.write_text(json.dumps(D), encoding="utf-8")
    r = correr("gate.py", "advance", "diagnostico", run=run, check=False)
    assert "declaracion.md" in r.stderr
    (run / "diagnostics" / "declaracion.md").write_text(DECLARACION, encoding="utf-8")
    correr("gate.py", "advance", "diagnostico", run=run, check=True)


def test_diagnostico_obsoleto_si_cambia_la_seleccion(run):
    agregar_decision(run, 7, "Override a B10", "Se elige B10 (Ridge) por simplicidad operativa ante el área usuaria.")
    correr("compare_models.py", "--override", "B10", "--justificacion", "Simplicidad operativa: el área usuaria mantiene modelos lineales.",
           run=run, check=True)
    correr("gate.py", "advance", "experimentos", run=run, check=True)
    r = correr("gate.py", "advance", "diagnostico", run=run, check=False)
    assert "otra comparación" in r.stderr
    correr("diagnostics.py", run=run, check=True)
    assert leer(run, "diagnostics/diagnostics.json")["seleccionado"] == "B10"
