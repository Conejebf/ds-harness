"""
Registro de métricas del harness.

Cada métrica sabe:
  * su tipo de problema (`reg` = regresión y forecasting, `clf` = clasificación),
  * si mayor es mejor,
  * cómo calcularse directamente, y
  * cuando es posible, cómo descomponerse en SUMAS por fila (`componentes` + `de_sumas`).

La descomposición permite que el bootstrap pareado de `compare_models.py` sea vectorizado:
se agregan componentes por unidad de remuestreo (bloque temporal, entidad o fila) y cada réplica
es un producto matriz-vector. Las métricas sin descomposición (roc_auc, pr_auc) se calculan
réplica por réplica (más lento).

Convenciones de clasificación: `clases` es la lista ordenada de clases; `P` es una matriz
(n, K) de probabilidades en ese orden; en binario la clase positiva es `clases[1]`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

EPS = 1e-15


@dataclass(frozen=True)
class Metrica:
    nombre: str
    tipo: str               # 'reg' | 'clf'
    mayor_mejor: bool
    requiere_proba: bool = False
    solo_binaria: bool = False
    componentes: Callable | None = None   # (y, yhat, P, K) -> (n, k)
    de_sumas: Callable | None = None      # (S[..., k]) -> valor[...]
    directa: Callable | None = None       # (y, yhat, P, K) -> float


def _div(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(b == 0, np.nan, a / np.where(b == 0, 1, b))


# ---------------------------------------------------------------- regresión
def _c_mae(y, yh, P, K):
    return np.column_stack([np.abs(y - yh), np.ones_like(y)])


def _c_mse(y, yh, P, K):
    return np.column_stack([(y - yh) ** 2, np.ones_like(y)])


def _c_wape(y, yh, P, K):
    return np.column_stack([np.abs(y - yh), np.abs(y)])


def _c_smape(y, yh, P, K):
    den = np.abs(y) + np.abs(yh)
    return np.column_stack([np.where(den == 0, 0.0, 2 * np.abs(y - yh) / np.where(den == 0, 1, den)), np.ones_like(y)])


def _c_r2(y, yh, P, K):
    return np.column_stack([(y - yh) ** 2, y, y ** 2, np.ones_like(y)])


def _s_r2(S):
    sst = S[..., 2] - _div(S[..., 1] ** 2, S[..., 3])
    return 1 - _div(S[..., 0], sst)


# ---------------------------------------------------------------- clasificación (y = índice de clase)
def _c_logloss(y, yh, P, K):
    p = np.clip(P[np.arange(len(y)), y.astype(int)], EPS, 1)
    return np.column_stack([-np.log(p), np.ones(len(y))])


def _c_brier(y, yh, P, K):
    return np.column_stack([(P[:, 1] - y) ** 2, np.ones(len(y))])


def _c_acc(y, yh, P, K):
    return np.column_stack([(y == yh).astype(float), np.ones(len(y))])


def _c_f1(y, yh, P, K):
    tp = ((yh == 1) & (y == 1)).astype(float)
    fp = ((yh == 1) & (y == 0)).astype(float)
    fn = ((yh == 0) & (y == 1)).astype(float)
    return np.column_stack([tp, fp, fn])


def _s_f1(S):
    return _div(2 * S[..., 0], 2 * S[..., 0] + S[..., 1] + S[..., 2])


def _c_f1_macro(y, yh, P, K):
    cols = []
    for k in range(K):
        cols += [((yh == k) & (y == k)), ((yh == k) & (y != k)), ((yh != k) & (y == k))]
    return np.column_stack(cols).astype(float)


def _s_f1_macro(S):
    K = S.shape[-1] // 3
    f = [_div(2 * S[..., 3 * k], 2 * S[..., 3 * k] + S[..., 3 * k + 1] + S[..., 3 * k + 2]) for k in range(K)]
    return np.nanmean(np.stack(f, axis=-1), axis=-1)


def _c_bacc(y, yh, P, K):
    cols = []
    for k in range(K):
        cols += [((yh == k) & (y == k)), (y == k)]
    return np.column_stack(cols).astype(float)


def _s_bacc(S):
    K = S.shape[-1] // 2
    r = [_div(S[..., 2 * k], S[..., 2 * k + 1]) for k in range(K)]
    return np.nanmean(np.stack(r, axis=-1), axis=-1)


def _d_auc(y, yh, P, K):
    from sklearn.metrics import roc_auc_score
    if len(np.unique(y)) < 2:
        return np.nan
    if K == 2:
        return float(roc_auc_score(y, P[:, 1]))
    return float(roc_auc_score(y, P, multi_class="ovr", average="macro", labels=list(range(K))))


def _d_prauc(y, yh, P, K):
    from sklearn.metrics import average_precision_score
    if len(np.unique(y)) < 2:
        return np.nan
    return float(average_precision_score(y, P[:, 1]))


def _ratio(S):
    return _div(S[..., 0], S[..., 1])


METRICAS: dict[str, Metrica] = {
    "mae": Metrica("mae", "reg", False, componentes=_c_mae, de_sumas=_ratio),
    "mse": Metrica("mse", "reg", False, componentes=_c_mse, de_sumas=_ratio),
    "rmse": Metrica("rmse", "reg", False, componentes=_c_mse, de_sumas=lambda S: np.sqrt(_ratio(S))),
    "wape": Metrica("wape", "reg", False, componentes=_c_wape, de_sumas=_ratio),
    "smape": Metrica("smape", "reg", False, componentes=_c_smape, de_sumas=_ratio),
    "r2": Metrica("r2", "reg", True, componentes=_c_r2, de_sumas=_s_r2),
    "log_loss": Metrica("log_loss", "clf", False, requiere_proba=True, componentes=_c_logloss, de_sumas=_ratio),
    "brier": Metrica("brier", "clf", False, requiere_proba=True, solo_binaria=True, componentes=_c_brier, de_sumas=_ratio),
    "accuracy": Metrica("accuracy", "clf", True, componentes=_c_acc, de_sumas=_ratio),
    "f1": Metrica("f1", "clf", True, solo_binaria=True, componentes=_c_f1, de_sumas=_s_f1),
    "f1_macro": Metrica("f1_macro", "clf", True, componentes=_c_f1_macro, de_sumas=_s_f1_macro),
    "balanced_accuracy": Metrica("balanced_accuracy", "clf", True, componentes=_c_bacc, de_sumas=_s_bacc),
    "roc_auc": Metrica("roc_auc", "clf", True, requiere_proba=True, directa=_d_auc),
    "pr_auc": Metrica("pr_auc", "clf", True, requiere_proba=True, solo_binaria=True, directa=_d_prauc),
}

TIPO_PROBLEMA = {"forecasting": "reg", "regression": "reg", "classification": "clf"}


def compatibles(problem_type: str, n_clases: int | None = None) -> list[str]:
    t = TIPO_PROBLEMA[problem_type]
    out = []
    for m in METRICAS.values():
        if m.tipo != t:
            continue
        if m.solo_binaria and n_clases is not None and n_clases > 2:
            continue
        out.append(m.nombre)
    return out


def preparar(nombre: str, y, yhat, P, clases):
    """Convierte etiquetas a índices para clasificación. Devuelve (y, yhat, P, K)."""
    m = METRICAS[nombre]
    if m.tipo == "reg":
        return np.asarray(y, float), np.asarray(yhat, float), None, 0
    idx = {c: i for i, c in enumerate(clases)}
    yi = np.array([idx[v] for v in y])
    yhi = np.array([idx.get(v, -1) for v in yhat])
    return yi, yhi, (None if P is None else np.asarray(P, float)), len(clases)


def calcular(nombre: str, y, yhat, P=None, clases=None) -> float:
    m = METRICAS[nombre]
    yy, yh, PP, K = preparar(nombre, y, yhat, P, clases)
    if len(yy) == 0:
        return float("nan")
    if m.componentes is not None:
        S = m.componentes(yy, yh, PP, K).sum(axis=0)
        return float(m.de_sumas(S))
    return float(m.directa(yy, yh, PP, K))
