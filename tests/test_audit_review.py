"""Fase 5: auditoría determinista, fuga que se escapa del EDA y reglas de la revisión."""
import json
import shutil
from pathlib import Path

import pytest

from conftest import correr
from experimentos import EXPERIMENTOS
from helpers import (agregar_decision, dictamen, editar_contrato, escribir_interpretacion, experimentar, excluir,
                     hasta_diagnostico, hasta_eda, justificar_senales, leer, ronda_revision)
from synth import crear_run


# --------------------------------------------------------------------------- fuga que se escapa

def _escapar_fuga(tmp_path, con_linaje: bool):
    """Un experimentador excluye la fuga obvia pero 'justifica' indice_demanda (ventas del día siguiente + ruido).
    Con linaje declarado (falso) la declara como rezago de 7 días."""
    run = crear_run(tmp_path / ("lin" if con_linaje else "sin"), "forecasting")
    if con_linaje:
        editar_contrato(run, lambda c: c["features"].update(indice_demanda={
            "disponibilidad": "con_retraso", "retraso": 7, "linaje": {"fuente": "ventas", "desfase": 7}}))
    correr("contract.py", "validate", run=run, check=True)
    excluir(run, ["ventas_ajustadas"])
    correr("contract.py", "validate", run=run, check=True)
    correr("profile_data.py", run=run, check=True)
    correr("leakage_check.py", run=run, check=False)
    assert any("indice_demanda" in i for i in justificar_senales(run))     # leakage_check SÍ la detectó...
    correr("leakage_check.py", run=run, check=True)                         # ...pero la "justificación" la deja pasar
    escribir_interpretacion(run)
    correr("gate.py", "advance", "eda", run=run, check=True)
    correr("baselines.py", run=run, check=True)
    correr("gate.py", "advance", "baselines", run=run, check=True)
    experimentar(run, "forecasting", ("E001",))
    correr("experiment_log.py", "stop", "--motivo", "saturacion", "--evidencia",
           "El baseline HGB con todas las features ya logra un MAE muy bajo; no hay hipótesis con mecanismo que lo mejore.",
           run=run, check=True)
    hasta_diagnostico(run)
    sel = leer(run, "comparison.json")["congelado"]
    assert "indice_demanda" in sel["features"]   # la fuga llegó al modelo elegido
    return run


def test_fuga_escapada_con_linaje_falso_es_bloqueante(tmp_path):
    run = _escapar_fuga(tmp_path, con_linaje=True)
    r = correr("audit.py", run=run, check=False)
    A = leer(run, "audit.json")
    chk = {x["id"]: x for x in A["verificaciones"]}["A-DISP-indice_demanda"]
    assert chk["veredicto"] == "falla" and chk["severidad"] == "bloqueante"
    assert "NO reproduce" in chk["evidencia"] and "futuro" in chk["evidencia"]
    assert {x["id"]: x for x in A["verificaciones"]}["A-LEAKAGE"]["veredicto"] == "falla"   # señal justificada a revisar
    assert "BLOQUEADO" in r.stdout
    # aunque el "revisor" no reporte nada, la falla de auditoría entra sola y bloquea
    ronda_revision(run, check_submit=False)
    R = leer(run, "review.json")
    assert R["hallazgos"]["A-DISP-indice_demanda"]["estado"] == "abierto"
    r = correr("gate.py", "advance", "revision", run=run, check=False)
    assert "bloqueantes abiertos" in r.stderr
    assert "'revision'" in correr("holdout.py", "evaluate", run=run, check=False).stderr
    assert "'revision'" in correr("final_report.py", run=run, check=False).stderr


def test_fuga_escapada_sin_linaje_es_no_verificable_mas_senal(tmp_path):
    run = _escapar_fuga(tmp_path, con_linaje=False)
    correr("audit.py", run=run)
    V = {x["id"]: x for x in leer(run, "audit.json")["verificaciones"]}
    assert V["A-DISP-indice_demanda"]["veredicto"] == "no_verificable"
    assert V["A-SENAL-indice_demanda"]["veredicto"] == "falla" and V["A-SENAL-indice_demanda"]["severidad"] == "importante"
    # no es bloqueante por sí sola: le toca al revisor (el test con LLM real comprueba que la escale)
    ronda_revision(run, check_submit=True)
    assert leer(run, "review.json")["hallazgos"]["A-SENAL-indice_demanda"]["estado"] == "abierto"


# --------------------------------------------------------------------------- run de regresión hasta diagnóstico

@pytest.fixture(scope="module")
def base_diag(tmp_path_factory):
    run = crear_run(tmp_path_factory.mktemp("diag") / "reg", "regression")
    hasta_eda(run, "regression")
    correr("baselines.py", run=run, check=True)
    correr("gate.py", "advance", "baselines", run=run, check=True)
    experimentar(run, "regression")
    hasta_diagnostico(run)
    return run


@pytest.fixture
def run(base_diag, tmp_path):
    dst = tmp_path / "reg"
    shutil.copytree(base_diag, dst)
    return dst


def test_auditoria_limpia(run):
    correr("audit.py", "--alcance", "todos", run=run, check=True)
    A = leer(run, "audit.json")
    V = {x["id"]: x["veredicto"] for x in A["verificaciones"]}
    for k in ("A-INTEGRIDAD", "A-SPLITS", "A-METRICAS", "A-REEJECUCION", "A-AISLAMIENTO", "A-HOLDOUT"):
        assert V[k] == "verificado", (k, A)
    assert A["bloqueantes"] == []


def test_auditoria_detecta_predicciones_alteradas(run):
    from common import read_table, write_table
    rec = [json.loads(l) for l in (run / "experiments.jsonl").read_text(encoding="utf-8").splitlines() if '"E002"' in l][0]
    p = run / rec["preds_ref"]
    df = read_table(p)
    df.loc[0, "y_pred"] += 5
    write_table(df, p)
    correr("audit.py", run=run, check=False)
    V = {x["id"]: x for x in leer(run, "audit.json")["verificaciones"]}
    assert V["A-INTEGRIDAD"]["veredicto"] == "falla" and V["A-METRICAS"]["veredicto"] == "falla"


def _nuevo_experimento(run, eid, codigo):
    (run / "experiments" / f"{eid}.py").write_text(codigo, encoding="utf-8")
    correr("experiment_log.py", "run", eid, run=run, check=True)
    correr("experiment_log.py", "conclude", eid, "--resultado", "Resultado observado documentado con cifras en experiments.jsonl",
           "--decision", "descartada", run=run, check=True)
    correr("compare_models.py", run=run, check=True)
    correr("gate.py", "advance", "experimentos", run=run, check=True)
    correr("diagnostics.py", run=run, check=True)
    correr("gate.py", "advance", "diagnostico", run=run, check=True)


def test_transformacion_ajustada_al_predecir_es_detectada(run):
    lotero = EXPERIMENTOS["regression"]["E004"].replace(
        "def build_pipeline(config, seed):",
        "from sklearn.base import BaseEstimator, TransformerMixin\n\n"
        "class EscalaDelLote(BaseEstimator, TransformerMixin):\n"
        "    def fit(self, X, y=None):\n        self.ok_ = True\n        return self\n"
        "    def transform(self, X):\n        X = X.copy()\n        num = X.select_dtypes('number').columns\n"
        "        X[num] = (X[num] - X[num].mean()) / X[num].std()\n        return X\n\n\n"
        "def build_pipeline(config, seed):"
    ).replace("return Pipeline([('pre', preprocesador_lineal()),", "return Pipeline([('lote', EscalaDelLote()), ('pre', preprocesador_lineal()),")
    lotero = lotero.replace("Solo tres variables importan", "Estandarizar con estadísticas del lote corrige el desplazamiento")
    _nuevo_experimento(run, "E006", lotero)
    correr("audit.py", run=run, check=False)
    V = {x["id"]: x for x in leer(run, "audit.json")["verificaciones"]}
    assert V["A-AISLAMIENTO"]["veredicto"] == "falla" and "E006" in V["A-AISLAMIENTO"]["evidencia"]


def test_experimento_que_carga_datos_es_detectado(run):
    codigo = EXPERIMENTOS["regression"]["E004"].replace(
        "    return Pipeline([('pre'",
        "    import pandas as pd\n    try:\n        _ = pd.read_csv('data/tabla.csv')\n    except OSError:\n        pass\n    return Pipeline([('pre'"
    ).replace("Solo tres variables importan", "Leer la tabla completa permite calibrar el escalado")
    _nuevo_experimento(run, "E006", codigo)
    correr("audit.py", run=run, check=False)
    V = {x["id"]: x for x in leer(run, "audit.json")["verificaciones"]}
    assert V["A-AISLAMIENTO"]["veredicto"] == "falla" and "read_csv" in V["A-AISLAMIENTO"]["evidencia"]


BLOQ = {"id": "R-001", "severidad": "bloqueante", "estado": "abierto", "evidencia": "El margen práctico no tiene respaldo de negocio",
        "artefacto": "contract.yaml", "accion_requerida": "documentar el margen en decisions.md"}


def test_reglas_de_la_revision(run):
    n, r = ronda_revision(run, [BLOQ], check_submit=False)
    assert "Bloqueantes abiertos: ['R-001']" in r.stdout
    assert "R-001" in correr("gate.py", "advance", "revision", run=run, check=False).stderr
    # ronda 2 sin cambiar nada: no se puede cerrar el bloqueante (el experimentador no se lo auto-cierra)
    correr("review.py", "prepare", run=run, check=True)
    resuelto = dict(BLOQ, estado="resuelto", resolucion="ya se discutió con el equipo y está bien")
    r = correr("review.py", "submit", "--file", dictamen(run, 2, [resuelto]), run=run, check=False)
    assert "sin que cambiara ningún artefacto" in r.stderr
    r = correr("review.py", "submit", "--file", dictamen(run, 2, []), run=run, check=False)
    assert "no se pronuncia" in r.stderr
    r = correr("review.py", "submit", "--file", dictamen(run, 2, [dict(BLOQ, estado="aceptado", resolucion="riesgo aceptado por el equipo")]),
               run=run, check=False)
    assert "no se puede aceptar" in r.stderr
    # corrección real + nueva ronda
    agregar_decision(run, 5, "Margen práctico", "El margen de 0.05 de RMSE es el error mínimo perceptible por el área usuaria.")
    r = correr("review.py", "submit", "--file", dictamen(run, 2, [resuelto]), run=run, check=False)
    assert "cambiaron después" in r.stderr
    n, r = ronda_revision(run, [dict(BLOQ, estado="resuelto", resolucion="D-005 documenta el margen con respaldo del área usuaria")],
                          check_submit=True)
    correr("gate.py", "advance", "revision", run=run, check=True)
    # cambiar artefactos después de la revisión la vuelve obsoleta
    agregar_decision(run, 6, "Otra", "Cambio posterior a la revisión.")
    r = correr("gate.py", "advance", "revision", run=run, check=False)
    assert "versión anterior" in r.stderr or "no vigente" in r.stderr


def test_dictamen_invalido_rechazado(run):
    correr("audit.py", run=run, check=True)
    correr("review.py", "prepare", run=run, check=True)
    p = dictamen(run, 1, [{"id": "X1", "severidad": "grave", "estado": "abierto"}])
    r = correr("review.py", "submit", "--file", p, run=run, check=False)
    assert "id inválido" in r.stderr and "severidad inválida" in r.stderr and "falta `evidencia`" in r.stderr
