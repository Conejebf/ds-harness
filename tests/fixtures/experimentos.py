"""Cinco hipótesis genuinamente distintas por tipo de problema (las escribe 'el experimentador' en las pruebas)."""
from pathlib import Path

CAB = '''from sklearn.pipeline import Pipeline
from modelos import preprocesador_arboles, preprocesador_lineal
'''


def _exp(hip, cat, ref, cambio, esperado, descarte, feats, familia, estoc, config, cuerpo, imports=""):
    return (CAB + imports + f'''
HIPOTESIS = {{"hipotesis": {hip!r}, "categoria": {cat!r}, "referencia": {ref!r}, "cambio": {cambio!r},
             "esperado": {esperado!r}, "criterio_descarte": {descarte!r}}}
FEATURES = {feats!r}
FAMILIA = {familia!r}
ESTOCASTICO = {estoc!r}
CONFIG = {config!r}


def build_pipeline(config, seed):
{cuerpo}
''')


F_ALL = ["tienda", "dow", "promo", "temp", "ventas_lag7", "ventas_lag14", "media_movil_7"]
C_ALL = ["region", "edad", "antiguedad_meses", "uso_mensual", "quejas"]
R_ALL = ["x1", "x2", "x3", "x4", "x5", "cat"]

EXPERIMENTOS = {
    "forecasting": {
        "E001": _exp("La temperatura casi no aporta y agrega ruido; quitarla reduce el error", "features", "B20",
                     "HGB sin temp", "MAE menor o igual a B20", "MAE peor que B20 en más del margen",
                     [f for f in F_ALL if f != "temp"], "gbm", False, {"max_iter": 200, "learning_rate": 0.05},
                     "    return Pipeline([('pre', preprocesador_arboles()), ('m', HistGradientBoostingRegressor(early_stopping=False, random_state=seed, **config))])",
                     "from sklearn.ensemble import HistGradientBoostingRegressor\n"),
        "E002": _exp("Un bosque aleatorio captura interacciones promo x tienda mejor que el boosting", "familia_modelo", "B20",
                     "RandomForest con hojas mínimas de 5", "MAE menor que B20", "MAE peor que B20",
                     F_ALL, "arbol", True, {"n_estimators": 120, "min_samples_leaf": 5},
                     "    return Pipeline([('pre', preprocesador_arboles()), ('m', RandomForestRegressor(random_state=seed, n_jobs=1, **config))])",
                     "from sklearn.ensemble import RandomForestRegressor\n"),
        "E003": _exp("El efecto de las variables es multiplicativo: modelar log(ventas) linealmente reduce el error", "transformacion_target", "B10",
                     "Ridge sobre log1p(ventas)", "MAE menor que B10", "MAE peor que B10",
                     F_ALL, "lineal", False, {"alphas": [0.1, 1.0, 10.0]},
                     "    return TransformedTargetRegressor(Pipeline([('pre', preprocesador_lineal()), ('m', RidgeCV(**config))]), func=np.log1p, inverse_func=np.expm1)",
                     "import numpy as np\nfrom sklearn.compose import TransformedTargetRegressor\nfrom sklearn.linear_model import RidgeCV\n"),
        "E004": _exp("El boosting sobreajusta con 1,000 filas; regularizarlo (profundidad 3, L2) mejora la generalización", "regularizacion", "B20",
                     "max_depth=3, l2=5", "MAE menor que B20", "MAE peor que B20",
                     F_ALL, "gbm", False, {"max_iter": 200, "learning_rate": 0.05, "max_depth": 3, "l2_regularization": 5.0},
                     "    return Pipeline([('pre', preprocesador_arboles()), ('m', HistGradientBoostingRegressor(early_stopping=False, random_state=seed, **config))])",
                     "from sklearn.ensemble import HistGradientBoostingRegressor\n"),
        "E005": _exp("Los picos de promoción dominan la pérdida cuadrática; entrenar con pérdida absoluta alinea el modelo con el MAE", "desbalance_outliers", "B20",
                     "loss=absolute_error", "MAE menor que B20", "MAE peor que B20",
                     F_ALL, "gbm", False, {"max_iter": 200, "learning_rate": 0.05, "loss": "absolute_error"},
                     "    return Pipeline([('pre', preprocesador_arboles()), ('m', HistGradientBoostingRegressor(early_stopping=False, random_state=seed, **config))])",
                     "from sklearn.ensemble import HistGradientBoostingRegressor\n"),
    },
    "classification": {
        "E001": _exp("La logística sobreajusta poco, pero más regularización mejora la calibración", "regularizacion", "B10",
                     "C=0.1", "log_loss menor que B10", "log_loss peor que B10",
                     C_ALL, "lineal", False, {"C": 0.1, "max_iter": 2000},
                     "    return Pipeline([('pre', preprocesador_lineal()), ('m', LogisticRegression(**config))])",
                     "from sklearn.linear_model import LogisticRegression\n"),
        "E002": _exp("Un bosque con hojas grandes da probabilidades suaves y captura no linealidades de uso", "familia_modelo", "B10",
                     "RandomForest min_samples_leaf=20", "log_loss menor que B10", "log_loss peor que B10",
                     C_ALL, "arbol", True, {"n_estimators": 150, "min_samples_leaf": 20},
                     "    return Pipeline([('pre', preprocesador_arboles()), ('m', RandomForestClassifier(random_state=seed, n_jobs=1, **config))])",
                     "from sklearn.ensemble import RandomForestClassifier\n"),
        "E003": _exp("El boosting base sobreajusta; con menos iteraciones y árboles poco profundos deja de hacerlo", "hiperparametros", "B20",
                     "max_iter=80, lr=0.03, max_depth=3, l2=1", "log_loss menor que B20 y cercano a B10", "log_loss peor que B20",
                     C_ALL, "gbm", False, {"max_iter": 80, "learning_rate": 0.03, "max_depth": 3, "l2_regularization": 1.0},
                     "    return Pipeline([('pre', preprocesador_arboles()), ('m', HistGradientBoostingClassifier(early_stopping=False, random_state=seed, **config))])",
                     "from sklearn.ensemble import HistGradientBoostingClassifier\n"),
        "E004": _exp("Edad y región no tienen relación causal con el churn; quitarlas no empeora", "features", "B10",
                     "logística solo con quejas, antigüedad y uso", "log_loss igual a B10 dentro del margen", "log_loss peor que B10 más que el margen",
                     ["antiguedad_meses", "uso_mensual", "quejas"], "lineal", False, {"C": 1.0, "max_iter": 2000},
                     "    return Pipeline([('pre', preprocesador_lineal()), ('m', LogisticRegression(**config))])",
                     "from sklearn.linear_model import LogisticRegression\n"),
        "E005": _exp("El desbalance sesga la frontera; ponderar clases mejora la detección de bajas", "desbalance_outliers", "B10",
                     "class_weight=balanced", "log_loss similar y mejor recall", "log_loss peor que B10 más que el margen",
                     C_ALL, "lineal", False, {"class_weight": "balanced", "max_iter": 2000},
                     "    return Pipeline([('pre', preprocesador_lineal()), ('m', LogisticRegression(**config))])",
                     "from sklearn.linear_model import LogisticRegression\n"),
    },
    "regression": {
        "E001": _exp("Hay una interacción x3*x4 que el modelo lineal no ve; agregar interacciones de orden 2 la captura", "features", "B10",
                     "Ridge con interacciones de orden 2", "RMSE menor que B10", "RMSE peor que B10",
                     R_ALL, "lineal", False, {"alphas": [0.1, 1.0, 10.0]},
                     "    pre = Pipeline([('base', preprocesador_lineal()), ('poly', PolynomialFeatures(2, interaction_only=True, include_bias=False))])\n    return Pipeline([('pre', pre), ('m', RidgeCV(**config))])",
                     "from sklearn.linear_model import RidgeCV\nfrom sklearn.preprocessing import PolynomialFeatures\n"),
        "E002": _exp("Un bosque aleatorio captura la no linealidad sin(x5) mejor que el boosting", "familia_modelo", "B20",
                     "RandomForest", "RMSE menor que B20", "RMSE peor que B20",
                     R_ALL, "arbol", True, {"n_estimators": 150, "min_samples_leaf": 3},
                     "    return Pipeline([('pre', preprocesador_arboles()), ('m', RandomForestRegressor(random_state=seed, n_jobs=1, **config))])",
                     "from sklearn.ensemble import RandomForestRegressor\n"),
        "E003": _exp("Con más iteraciones y tasa mayor el boosting reduce el sesgo", "hiperparametros", "B20",
                     "max_iter=400, lr=0.1", "RMSE menor que B20", "RMSE peor que B20",
                     R_ALL, "gbm", False, {"max_iter": 400, "learning_rate": 0.1},
                     "    return Pipeline([('pre', preprocesador_arboles()), ('m', HistGradientBoostingRegressor(early_stopping=False, random_state=seed, **config))])",
                     "from sklearn.ensemble import HistGradientBoostingRegressor\n"),
        "E004": _exp("Solo tres variables importan; la regularización L1 elimina el resto y reduce varianza", "regularizacion", "B10",
                     "LassoCV", "RMSE menor o igual que B10", "RMSE peor que B10",
                     R_ALL, "lineal", False, {"cv": 3},
                     "    return Pipeline([('pre', preprocesador_lineal()), ('m', LassoCV(random_state=seed, **config))])",
                     "from sklearn.linear_model import LassoCV\n"),
        "E005": _exp("La relación es local y suave; un modelo de vecinos cercanos sobre variables escaladas compite con el boosting", "otra", "B20",
                     "KNN k=15 con pesos por distancia", "RMSE cercano a B20", "RMSE peor que B10",
                     R_ALL, "otro", False, {"n_neighbors": 15, "weights": "distance"},
                     "    return Pipeline([('pre', preprocesador_lineal()), ('m', KNeighborsRegressor(**config))])",
                     "from sklearn.neighbors import KNeighborsRegressor\n"),
    },
}


def instalar(run, tipo, ids=None):
    for k, code in EXPERIMENTOS[tipo].items():
        if ids is None or k in ids:
            (Path(run) / "experiments" / f"{k}.py").write_text(code, encoding="utf-8")
