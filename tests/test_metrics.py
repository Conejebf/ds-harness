"""Las métricas descompuestas en sumas coinciden con scikit-learn (binario, multiclase y regresión)."""
import numpy as np
import pytest
from sklearn import metrics as skm

from metrics import METRICAS, calcular, compatibles

rng = np.random.RandomState(0)


def test_regresion_vs_sklearn():
    y, yh = rng.normal(size=200), rng.normal(size=200)
    assert np.isclose(calcular("mae", y, yh), skm.mean_absolute_error(y, yh))
    assert np.isclose(calcular("rmse", y, yh), np.sqrt(skm.mean_squared_error(y, yh)))
    assert np.isclose(calcular("r2", y, yh), skm.r2_score(y, yh))
    assert np.isclose(calcular("wape", y, yh), np.abs(y - yh).sum() / np.abs(y).sum())


def test_binaria_vs_sklearn():
    y = rng.randint(0, 2, 300)
    p = np.clip(rng.rand(300) * 0.6 + 0.4 * y, 0, 1)
    P = np.column_stack([1 - p, p])
    yh = (p >= 0.5).astype(int)
    cl = [0, 1]
    assert np.isclose(calcular("log_loss", y, yh, P, cl), skm.log_loss(y, P))
    assert np.isclose(calcular("roc_auc", y, yh, P, cl), skm.roc_auc_score(y, p))
    assert np.isclose(calcular("f1", y, yh, P, cl), skm.f1_score(y, yh))
    assert np.isclose(calcular("brier", y, yh, P, cl), skm.brier_score_loss(y, p))
    assert np.isclose(calcular("balanced_accuracy", y, yh, P, cl), skm.balanced_accuracy_score(y, yh))


def test_multiclase_vs_sklearn():
    cl = ["a", "b", "c"]
    y = rng.choice(cl, 300)
    P = rng.dirichlet([1, 1, 1], 300)
    yh = np.array(cl)[P.argmax(1)]
    assert np.isclose(calcular("log_loss", y, yh, P, cl), skm.log_loss(y, P, labels=cl))
    assert np.isclose(calcular("f1_macro", y, yh, P, cl), skm.f1_score(y, yh, average="macro"))
    assert np.isclose(calcular("accuracy", y, yh, P, cl), skm.accuracy_score(y, yh))
    assert np.isclose(calcular("roc_auc", y, yh, P, cl), skm.roc_auc_score(y, P, multi_class="ovr", labels=cl))
    assert "f1" not in compatibles("classification", 3) and "brier" not in compatibles("classification", 3)


@pytest.mark.parametrize("nombre", [k for k, m in METRICAS.items() if m.componentes is not None and m.tipo == "reg"])
def test_sumas_son_aditivas(nombre):
    """La descomposición por sumas es la base del bootstrap vectorizado."""
    m = METRICAS[nombre]
    y, yh = rng.normal(size=100) + 5, rng.normal(size=100) + 5
    S = m.componentes(y, yh, None, 0)
    assert np.isclose(m.de_sumas(S[:60].sum(0) + S[60:].sum(0)), calcular(nombre, y, yh))
