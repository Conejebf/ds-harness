"""
Plantilla de experimento. Copia a experiments/E0xx.py (xx = número consecutivo) y llena.

Reglas:
  * Todo lo que aprenda de los datos (imputación, escalado, codificación, selección de variables,
    remuestreo, transformación del target) va DENTRO del pipeline: el harness lo ajusta solo con
    el entrenamiento de cada fold.
  * No cargues datos aquí: el harness pasa X e y. Leer archivos desde el experimento es una
    violación que detecta audit.py.
  * `build_pipeline` debe devolver un estimador SIN ajustar.
  * No escribas métricas: `experiment_log.py run` las calcula.
"""
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline

HIPOTESIS = {
    "hipotesis": "Qué crees y por qué (una frase con mecanismo, no 'probar X').",
    "categoria": "features",          # features | familia_modelo | desbalance_outliers | transformacion_target | regularizacion | hiperparametros | otra
    "referencia": "B03",              # experimento o baseline contra el que se compara
    "cambio": "Qué cambia exactamente respecto de la referencia.",
    "esperado": "Qué resultado esperas si la hipótesis es cierta.",
    "criterio_descarte": "Qué resultado te haría descartarla.",
}
FEATURES = ["col_a", "col_b"]
FAMILIA = "gbm"                       # una de comparacion.orden_simplicidad del contrato
ESTOCASTICO = True                    # True => se corre con todas las semillas del contrato
CONFIG = {"max_iter": 200, "learning_rate": 0.05}


def build_pipeline(config, seed):
    pre = ColumnTransformer([("num", "passthrough", FEATURES)])
    return Pipeline([("pre", pre), ("modelo", HistGradientBoostingRegressor(random_state=seed, **config))])
