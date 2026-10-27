"""Fases 2-3: baselines, runner de experimentos, bitácora y comparación."""
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conftest import correr
from experimentos import EXPERIMENTOS, instalar
from helpers import agregar_decision, editar_contrato, hasta_eda
from synth import crear_run


@pytest.fixture(scope="module")
def base_reg(tmp_path_factory):
    """Run de regresión con contrato, EDA (fuga excluida) y baselines aprobados."""
    run = crear_run(tmp_path_factory.mktemp("base") / "reg", "regression")
    hasta_eda(run, "regression")
    correr("baselines.py", run=run, check=True)
    correr("gate.py", "advance", "baselines", run=run, check=True)
    return run


@pytest.fixture
def run(base_reg, tmp_path):
    dst = tmp_path / "reg"
    shutil.copytree(base_reg, dst)
    return dst


def _bitacora(run):
    return [json.loads(l) for l in (Path(run) / "experiments.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]


def _escribir(run, eid, codigo):
    (Path(run) / "experiments" / f"{eid}.py").write_text(codigo, encoding="utf-8")


def test_experimentos_bloqueados_sin_baselines(tmp_path):
    r = crear_run(tmp_path / "r", "regression")
    hasta_eda(r, "regression")
    instalar(r, "regression", ["E001"])
    out = correr("experiment_log.py", "run", "E001", run=r, check=False)
    assert "'baselines'" in out.stderr and "Cómo resolverlo" in out.stderr


def test_baselines_registrados(base_reg):
    B = json.loads((base_reg / "baselines.json").read_text(encoding="utf-8"))
    assert {b["familia"] for b in B["baselines"]} == {"ingenuo", "lineal", "gbm"}
    regs = [r for r in _bitacora(base_reg) if r["tipo"] == "baseline"]
    assert all(r["estado"] == "ok" and len(r["semillas"]) == 1 for r in regs)


def test_run_registra_metricas_preds_y_codigo(run):
    instalar(run, "regression", ["E002"])
    correr("experiment_log.py", "run", "E002", run=run, check=True)
    rec = [r for r in _bitacora(run) if r.get("id") == "E002"][-1]
    assert rec["estado"] == "ok" and rec["semillas"] == [0, 1, 2]                 # estocástico: 3 semillas
    assert len(rec["metricas_por_fold"]) == 10 * 3
    assert rec["resumen"]["rmse"]["std_entre_folds"] is not None
    assert (run / rec["codigo_ref"]).read_text(encoding="utf-8") == EXPERIMENTOS["regression"]["E002"]
    from common import read_table
    preds = read_table(run / rec["preds_ref"])
    assert set(preds.columns) >= {"id", "fold", "semilla", "y_true", "y_pred"}
    sp = json.loads((run / "splits.json").read_text(encoding="utf-8"))
    for f in sp["folds"]:
        assert set(preds.loc[(preds.fold == f["id"]) & (preds.semilla == 0), "id"]) == set(f["val"])
    assert rec["entorno"]["paquetes"]["sklearn"]


def test_determinista_una_semilla_y_reproduccion(run):
    instalar(run, "regression", ["E001"])
    correr("experiment_log.py", "run", "E001", run=run, check=True)
    r = correr("experiment_log.py", "run", "E001", run=run, check=False)
    assert "ya se ejecutó" in r.stderr
    correr("experiment_log.py", "run", "E001", "--repetir", run=run, check=True)
    regs = [r for r in _bitacora(run) if r.get("id") == "E001"]
    assert regs[0]["semillas"] == [0] and regs[1]["reproduccion_de"] == "E001"
    assert np.isclose(regs[0]["resumen"]["rmse"]["valor"], regs[1]["resumen"]["rmse"]["valor"], rtol=1e-9)


def test_no_se_sobrescribe_experimento_ejecutado(run):
    instalar(run, "regression", ["E001"])
    correr("experiment_log.py", "run", "E001", run=run, check=True)
    _escribir(run, "E001", EXPERIMENTOS["regression"]["E001"].replace("0.1, 1.0, 10.0", "0.5, 5.0"))
    r = correr("experiment_log.py", "run", "E001", run=run, check=False)
    assert "código cambió" in r.stderr and "id nuevo" in r.stderr


def test_fallos_se_registran(run):
    malo = EXPERIMENTOS["regression"]["E003"].replace(
        "def build_pipeline(config, seed):",
        "class Roto(Pipeline):\n    def predict(self, X):\n        return super().predict(X)[:-1]\n\n\ndef build_pipeline(config, seed):"
    ).replace("return Pipeline(", "return Roto(")
    _escribir(run, "E003", malo)
    r = correr("experiment_log.py", "run", "E003", run=run, check=False)
    assert "FALLÓ" in r.stdout and "desalineadas" in r.stdout
    rec = [x for x in _bitacora(run) if x.get("id") == "E003"][-1]
    assert rec["estado"] == "fallido" and "desalineadas" in rec["error"]


def test_pipeline_preajustado_es_rechazado(run):
    codigo = EXPERIMENTOS["regression"]["E004"].replace(
        "    return Pipeline([('pre', preprocesador_lineal()), ('m', LassoCV(random_state=seed, **config))])",
        "    import numpy as np, pandas as pd\n    p = Pipeline([('pre', preprocesador_lineal()), ('m', LassoCV(random_state=seed, **config))])\n"
        "    X = pd.DataFrame({k: np.random.rand(20) for k in FEATURES[:-1]}); X['cat'] = 'a'\n"
        "    return p.fit(X, np.random.rand(20))")
    _escribir(run, "E004", codigo)
    correr("experiment_log.py", "run", "E004", run=run, check=False)
    rec = [x for x in _bitacora(run) if x.get("id") == "E004"][-1]
    assert rec["estado"] == "fallido" and "YA AJUSTADO" in rec["error"]


def test_features_excluidas_rechazadas(run):
    _escribir(run, "E005", EXPERIMENTOS["regression"]["E005"].replace("FEATURES = [", "FEATURES = ['y_copia_ruidosa', "))
    r = correr("experiment_log.py", "run", "E005", run=run, check=False)
    assert "no permitidas" in r.stderr and "y_copia_ruidosa" in r.stderr


def _correr_todos(run, ids=("E001", "E002", "E003", "E004", "E005"), concluir=True):
    instalar(run, "regression", list(ids))
    for e in ids:
        correr("experiment_log.py", "run", e, run=run, check=True)
        if concluir:
            correr("experiment_log.py", "conclude", e, "--resultado", "Se observó un cambio de RMSE consistente con lo registrado en experiments.jsonl",
                   "--decision", "descartada", run=run, check=True)


def test_gate_exploracion(run):
    _correr_todos(run, ("E001", "E002", "E003"), concluir=False)
    correr("compare_models.py", run=run, check=True)
    r = correr("gate.py", "advance", "experimentos", run=run, check=False)
    assert "3 hipótesis distintas" in r.stderr and "sin resultado observado" in r.stderr
    for e in ("E001", "E002", "E003"):
        correr("experiment_log.py", "conclude", e, "--resultado", "Resultado observado documentado con cifras de experiments.jsonl",
               "--decision", "aceptada", run=run, check=True)
    correr("experiment_log.py", "stop", "--motivo", "saturacion", "--evidencia", "x", run=run, check=False)
    correr("experiment_log.py", "stop", "--motivo", "saturacion", "--evidencia",
           "Las tres familias razonables quedan a < 0.05 de RMSE entre sí; no hay hipótesis nuevas con mecanismo plausible.",
           run=run, check=True)
    r = correr("gate.py", "advance", "experimentos", run=run, check=False)
    assert "desactualizado" in r.stderr    # el cierre cambió experiments.jsonl
    correr("compare_models.py", run=run, check=True)
    correr("gate.py", "advance", "experimentos", run=run, check=True)


def test_hipotesis_repetida_no_cuenta(run):
    _correr_todos(run, ("E001", "E002", "E003", "E004"))
    dup = EXPERIMENTOS["regression"]["E005"]
    import re
    texto = re.search(r'"hipotesis": (\'[^\']*\'|"[^"]*")', EXPERIMENTOS["regression"]["E002"]).group(1)
    dup = re.sub(r'"hipotesis": (\'[^\']*\'|"[^"]*")', lambda _: f'"hipotesis": {texto}', dup, count=1)
    _escribir(run, "E005", dup)
    r = correr("experiment_log.py", "run", "E005", run=run, check=True)
    assert "repite textualmente" in r.stdout
    correr("experiment_log.py", "conclude", "E005", "--resultado", "Resultado observado documentado con cifras de experiments.jsonl",
           "--decision", "descartada", run=run, check=True)
    correr("compare_models.py", run=run, check=True)
    r = correr("gate.py", "advance", "experimentos", run=run, check=False)
    assert "4 hipótesis distintas" in r.stderr and "E005" in r.stderr


def test_comparacion_completa_y_gate(run):
    _correr_todos(run)
    correr("compare_models.py", run=run, check=True)
    out = json.loads((run / "comparison.json").read_text(encoding="utf-8"))
    assert out["metodo"] == "nadeau_bengio" and out["inferencia_valida"]
    assert out["seleccion"]["estado"] in ("concluyente", "provisional")
    assert out["vs_ingenuo"]["p_no_mejora"] < 0.05
    assert all("ic" in x for x in out["candidatos"] if x["elegible"])
    for x in out["candidatos"]:
        if x["elegible"]:   # IC de la propia métrica (Nadeau–Bengio) que contiene la estimación puntual
            assert x["ic_metrica"][0] <= x["valor"] <= x["ic_metrica"][1]
    correr("gate.py", "advance", "experimentos", run=run, check=True)


def test_sin_inferencia_valida_la_eleccion_es_operativa(run):
    _correr_todos(run)
    editar_contrato(run, lambda c: c["comparacion"].update(min_folds_nb=20))
    from compare_models import comparar
    out = comparar(run)
    assert not out["inferencia_valida"] and out["seleccion"]["estado"] == "operativa"
    assert "SIN inferencia" in out["seleccion"]["criterio"]
    assert all(x["veredicto"] in ("mejor", "sin_inferencia") for x in out["candidatos"] if x["elegible"])


def test_cambiar_evaluacion_deja_obsoletos_y_obliga_a_rehacer(run):
    _correr_todos(run, ("E001",))
    editar_contrato(run, lambda c: c["validacion"].update(n_folds=4))
    correr("contract.py", "validate", run=run, check=True)
    assert "obsoleto" in correr("experiment_log.py", "list", run=run, check=True).stdout
    r = correr("compare_models.py", run=run, check=False)
    assert "'eda'" in r.stderr


def test_holm_y_nadeau_bengio():
    from compare_models import holm, nadeau_bengio
    assert np.allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    r = nadeau_bengio(np.array([0.1, 0.12, 0.09, 0.11, 0.1]), 0.25, 0.5, 0.05)
    assert r["p_noinf"] < 0.01 and r["ic"][0] < 0.1 < r["ic"][1]
    r = nadeau_bengio(np.zeros(5), 0.25, 0.0, 0.05)
    assert r["p_noinf"] == 1.0   # deg = margen = 0: no se puede afirmar no inferioridad estricta


def test_override_exige_decision(run):
    _correr_todos(run)
    r = correr("compare_models.py", "--override", "E005", "--justificacion", "El negocio exige un modelo explicable por vecinos.", run=run, check=False)
    assert "decisions.md" in r.stderr
    agregar_decision(run, 9, "Override a E005", "Se elige E005 (KNN) por explicabilidad ante el área usuaria.")
    correr("compare_models.py", "--override", "E005", "--justificacion", "El negocio exige un modelo explicable por vecinos.", run=run, check=True)
    out = json.loads((run / "comparison.json").read_text(encoding="utf-8"))
    assert out["seleccion"]["estado"] == "override" and out["seleccion"]["id"] == "E005"


def test_restriccion_excluye_candidatos(run):
    _correr_todos(run)
    editar_contrato(run, lambda c: c["restricciones"].update(familias_permitidas=["lineal"]))
    correr("contract.py", "validate", run=run, check=True)   # restricciones no cambian la evaluación
    from compare_models import comparar
    out = comparar(run)
    no = {x["id"] for x in out["candidatos"] if not x["elegible"]}
    assert {"E002", "E003", "E005"} <= no
    assert out["seleccion"]["id"] in ("E001", "E004", "B10")


def test_bootstrap_pareado_identicos_da_cero():
    from compare_models import Bootstrap
    rng = np.random.RandomState(0)
    p = pd.DataFrame({"fold": np.repeat(["r0f0", "r0f1"], 50), "entidad": np.tile(np.arange(25), 4).astype(str),
                      "semilla": 0, "y_true": rng.rand(100), "y_pred": rng.rand(100)})
    bs = Bootstrap(p, "grupos", 1, 200, {"r0f0": 0, "r0f1": 0})
    d1 = bs.distribucion(p, "mae", None, "media_folds")
    d2 = bs.distribucion(p.copy(), "mae", None, "media_folds")
    assert np.allclose(d1 - d2, 0) and d1.std() > 0
