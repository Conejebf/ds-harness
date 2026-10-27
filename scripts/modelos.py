"""
Bloques reutilizables para baselines y experimentos.

  preprocesador_lineal()   imputación + escalado (numéricas) e imputación + one-hot (categóricas)
  preprocesador_arboles()  codificación ordinal de categóricas (desconocidas/faltantes = -1) + numéricas tal cual
  NaiveUltimo, EstacionalIngenuo, MediaEntidad
                           baselines ingenuos de forecasting; requieren contexto (`_entidad`, `_t`),
                           que el runner agrega a X cuando el estimador tiene `requiere_contexto = True`.

Todo se ajusta dentro del fold: el runner clona/construye el pipeline por fold y solo le pasa
las filas de entrenamiento.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

NUM = make_column_selector(dtype_include="number")
CAT = make_column_selector(dtype_exclude="number")


def preprocesador_lineal() -> ColumnTransformer:
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("esc", StandardScaler())]), NUM),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), CAT),
    ])


def preprocesador_arboles() -> ColumnTransformer:
    return ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-1), CAT),
        ("num", "passthrough", NUM),
    ])


class _Contexto(BaseEstimator, RegressorMixin):
    requiere_contexto = True

    @staticmethod
    def _ent(X):
        return X["_entidad"].astype(str).values if "_entidad" in X else np.array(["_"] * len(X))


class NaiveUltimo(_Contexto):
    """Predice el último valor observado de la entidad en entrenamiento."""

    def fit(self, X, y):
        d = pd.DataFrame({"e": self._ent(X), "t": X["_t"].values, "y": np.asarray(y, float)})
        u = d.sort_values("t").groupby("e").tail(1)
        self.ultimo_ = dict(zip(u["e"], u["y"]))
        self.global_ = float(d.sort_values("t")["y"].iloc[-1])
        return self

    def predict(self, X):
        return np.array([self.ultimo_.get(e, self.global_) for e in self._ent(X)], float)


class EstacionalIngenuo(_Contexto):
    """y_hat(t) = y(t - k*m) con el menor k tal que t - k*m ya se observó en entrenamiento."""

    def __init__(self, m: int = 7):
        self.m = m

    def fit(self, X, y):
        d = pd.DataFrame({"e": self._ent(X), "t": X["_t"].values, "y": np.asarray(y, float)})
        self.hist_ = {(e, int(t)): v for e, t, v in zip(d["e"], d["t"], d["y"])}
        self.tmax_ = d.groupby("e")["t"].max().to_dict()
        self.ultimo_ = NaiveUltimo().fit(X, y)
        return self

    def predict(self, X):
        out = []
        base = self.ultimo_.predict(X)
        for i, (e, t) in enumerate(zip(self._ent(X), X["_t"].values)):
            tm = self.tmax_.get(e)
            v = None
            if tm is not None:
                k = max(1, math.ceil((int(t) - tm) / self.m))
                v = self.hist_.get((e, int(t) - k * self.m))
            out.append(base[i] if v is None else v)
        return np.array(out, float)


class MediaEntidad(_Contexto):
    """Media histórica de la entidad en entrenamiento."""

    def fit(self, X, y):
        d = pd.DataFrame({"e": self._ent(X), "y": np.asarray(y, float)})
        self.media_ = d.groupby("e")["y"].mean().to_dict()
        self.global_ = float(d["y"].mean())
        return self

    def predict(self, X):
        return np.array([self.media_.get(e, self.global_) for e in self._ent(X)], float)
