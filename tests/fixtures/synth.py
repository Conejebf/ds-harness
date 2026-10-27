"""
Datasets sintéticos pequeños para probar el harness, con fugas plantadas a propósito.

forecasting  8 tiendas x 200 días. Tendencia + estacionalidad semanal + promociones.
             `temp` es el pronóstico de temperatura (publicado al cierre de t), no la temperatura real.
             Fugas plantadas:
               * ventas_ajustadas = ventas * 1.02 + ruido      (derivada del target actual)
               * indice_demanda   = ventas del DÍA SIGUIENTE + ruido   (información futura disfrazada)
             Además: 10 filas duplicadas con otro row_id (duplicados accidentales).
classification  150 clientes x 10 filas. Generaliza a clientes nuevos (CV por grupos).
             Fuga: motivo_baja_registrado (casi igual al target).
regression   600 filas iid. Fuga: y_copia_ruidosa = y + ruido pequeño.

`con_fuga=False` produce la versión limpia (sin columnas de fuga ni duplicados).

CLI (crea un run de demostración):
    python tests/fixtures/synth.py --tipo forecasting --out runs/demo_fcst
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

TEMPLATES = Path(__file__).resolve().parents[2] / "scripts" / "templates"


def forecasting(con_fuga=True, seed=7, n_tiendas=8, n_dias=200) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    fechas = pd.date_range("2025-01-01", periods=n_dias, freq="D")
    filas = []
    for s in range(n_tiendas):
        base = rng.uniform(60, 100)
        t = np.arange(n_dias)
        dow = fechas.dayofweek.values
        promo = (rng.rand(n_dias) < 0.15).astype(int)
        temp = 20 + 6 * np.sin(2 * np.pi * t / 365) + rng.normal(0, 2, n_dias)
        y = base + 0.05 * t + 10 * np.sin(2 * np.pi * dow / 7) + 15 * promo + 0.4 * (temp - 20) + rng.normal(0, 8, n_dias)
        # `temp` es el PRONÓSTICO de temperatura para el día objetivo publicado al cierre del día t
        # (la real más error de pronóstico): la temperatura real de t+1..t+7 no se conoce al predecir.
        temp_pron = temp + rng.normal(0, 1.5, n_dias)
        df = pd.DataFrame({"fecha": fechas.strftime("%Y-%m-%d"), "tienda": f"T{s:02d}", "dow": dow, "promo": promo,
                           "temp": temp_pron.round(2), "ventas": y.round(3)})
        df["ventas_lag7"] = df["ventas"].shift(7)
        df["ventas_lag14"] = df["ventas"].shift(14)
        df["media_movil_7"] = df["ventas"].shift(7).rolling(7).mean().round(3)
        if con_fuga:
            df["ventas_ajustadas"] = (df["ventas"] * 1.02 + rng.normal(0, 0.5, n_dias)).round(3)
            df["indice_demanda"] = (df["ventas"].shift(-1) + rng.normal(0, 1, n_dias)).round(3)
        filas.append(df)
    out = pd.concat(filas, ignore_index=True)
    out.insert(0, "row_id", [f"F{i:05d}" for i in range(len(out))])
    if con_fuga:
        dups = out.sample(10, random_state=seed).copy()
        dups["row_id"] = [f"D{i:05d}" for i in range(len(dups))]
        out = pd.concat([out, dups], ignore_index=True)
    return out


def classification(con_fuga=True, seed=11, n_clientes=150, por_cliente=10) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    n = n_clientes * por_cliente
    cliente = np.repeat([f"C{i:03d}" for i in range(n_clientes)], por_cliente)
    efecto = np.repeat(rng.normal(0, 0.8, n_clientes), por_cliente)
    region = np.repeat(rng.choice(["norte", "centro", "sur"], n_clientes), por_cliente)
    edad = rng.randint(18, 70, n)
    antig = rng.exponential(24, n).round(1)
    uso = rng.gamma(2, 10, n).round(2)
    quejas = rng.poisson(0.7, n)
    logit = -1.0 + 0.6 * quejas - 0.03 * antig - 0.04 * (uso - 20) + efecto + 0.4 * (region == "sur")
    p = 1 / (1 + np.exp(-logit))
    y = (rng.rand(n) < p).astype(int)
    df = pd.DataFrame({"row_id": [f"R{i:05d}" for i in range(n)], "cliente": cliente, "region": region, "edad": edad,
                       "antiguedad_meses": antig, "uso_mensual": uso, "quejas": quejas, "churn": y})
    if con_fuga:
        df["motivo_baja_registrado"] = np.where(y == 1, (rng.rand(n) < 0.98).astype(int), 0)
    return df


def regression(con_fuga=True, seed=3, n=600) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    X = rng.normal(size=(n, 5))
    cat = rng.choice(["a", "b", "c"], n)
    y = 3 * X[:, 0] - 2 * X[:, 1] + 1.5 * X[:, 2] * X[:, 3] + np.sin(2 * X[:, 4]) + 2 * (cat == "c") + rng.normal(0, 1.0, n)
    df = pd.DataFrame(X.round(4), columns=[f"x{i}" for i in range(1, 6)])
    df["cat"] = cat
    df["y"] = y.round(4)
    df.insert(0, "row_id", [f"G{i:04d}" for i in range(n)])
    if con_fuga:
        df["y_copia_ruidosa"] = (y + rng.normal(0, 0.05 * y.std(), n)).round(4)
    return df


# ------------------------------------------------------------------------- contratos

BASE = yaml.safe_load((TEMPLATES / "contract.yaml").read_text(encoding="utf-8"))


def _contrato(**over) -> dict:
    c = copy.deepcopy(BASE)
    for ruta, v in over.items():
        cur = c
        partes = ruta.split("__")
        for k in partes[:-1]:
            cur = cur.setdefault(k, {})
        cur[partes[-1]] = v
    return c


def contrato_forecasting(con_fuga=True) -> dict:
    feats = {
        "tienda": {"disponibilidad": "conocida_anticipada"},
        "dow": {"disponibilidad": "conocida_anticipada"},
        "promo": {"disponibilidad": "conocida_anticipada"},
        "temp": {"disponibilidad": "al_predecir",
                 "descripcion": "pronóstico de temperatura para el día objetivo, publicado por el proveedor al cierre del día t"},
        "ventas_lag7": {"disponibilidad": "con_retraso", "retraso": 7, "linaje": {"fuente": "ventas", "desfase": 7}},
        "ventas_lag14": {"disponibilidad": "con_retraso", "retraso": 14, "linaje": {"fuente": "ventas", "desfase": 14}},
        "media_movil_7": {"disponibilidad": "con_retraso", "retraso": 7},
    }
    if con_fuga:  # el "analista" las declaró mal, como pasaría en la vida real
        feats["ventas_ajustadas"] = {"disponibilidad": "al_predecir"}
        feats["indice_demanda"] = {"disponibilidad": "al_predecir"}
    return _contrato(
        estudio="ventas_diarias_tiendas", problem_type="forecasting",
        datos__ruta="data/ventas.csv", datos__procedencia="Sintético (tests/fixtures/synth.py, semilla 7)",
        datos__id_fila="row_id", datos__target="ventas", datos__unidad_observacion="tienda-día",
        datos__entidad="tienda", datos__tiempo="fecha", datos__duplicados="accidentales",
        prediccion__momento="cierre del día t, para los días t+1..t+7", prediccion__horizonte=7,
        prediccion__retraso_target=0, prediccion__generalizacion="periodos_futuros", prediccion__estacionalidad=7,
        features=feats,
        validacion__esquema="temporal", validacion__n_folds=8, validacion__tamano_validacion=7, validacion__gap=0,
        validacion__justificacion="Se predice el futuro: backtest con origen móvil; gap 0 porque la venta se conoce al cierre.",
        holdout__estrategia="temporal_final", holdout__periodos=28,
        metrica__principal="mae", metrica__secundarias=["rmse", "wape"],
        comparacion__margen_practico=0.3, comparacion__n_boot=300,
        diagnostico__segmentos=["tienda", "promo"],
        diagnostico__grupos_features={"rezagos": ["ventas_lag7", "ventas_lag14", "media_movil_7"], "calendario": ["dow", "promo"]},
    )


def contrato_classification(con_fuga=True) -> dict:
    feats = {k: {"disponibilidad": "al_predecir"} for k in ("region", "edad", "antiguedad_meses", "uso_mensual", "quejas")}
    if con_fuga:
        feats["motivo_baja_registrado"] = {"disponibilidad": "al_predecir"}
    return _contrato(
        estudio="churn_clientes", problem_type="classification",
        datos__ruta="data/churn.csv", datos__procedencia="Sintético (tests/fixtures/synth.py, semilla 11)",
        datos__id_fila="row_id", datos__target="churn", datos__unidad_observacion="cliente-mes",
        datos__entidad="cliente", datos__columnas_excluidas=["cliente"],
        prediccion__momento="inicio de mes, con datos del mes anterior", prediccion__generalizacion="entidades_nuevas",
        features=feats,
        validacion__esquema="grupos", validacion__n_folds=5,
        validacion__justificacion="Se predice para clientes nuevos: ningún cliente puede estar en train y validación.",
        holdout__estrategia="grupos", holdout__fraccion=0.2,
        metrica__principal="log_loss", metrica__secundarias=["roc_auc", "brier"],
        comparacion__margen_practico=0.01, comparacion__n_boot=300,
        diagnostico__segmentos=["region"],
        diagnostico__grupos_features={"uso": ["uso_mensual", "quejas"], "perfil": ["edad", "antiguedad_meses", "region"]},
    )


def contrato_regression(con_fuga=True) -> dict:
    feats = {k: {"disponibilidad": "al_predecir"} for k in ("x1", "x2", "x3", "x4", "x5", "cat")}
    if con_fuga:
        feats["y_copia_ruidosa"] = {"disponibilidad": "al_predecir"}
    return _contrato(
        estudio="regresion_tabular", problem_type="regression",
        datos__ruta="data/tabla.csv", datos__procedencia="Sintético (tests/fixtures/synth.py, semilla 3)",
        datos__id_fila="row_id", datos__target="y", datos__unidad_observacion="registro",
        prediccion__momento="al registrar la observación", prediccion__generalizacion="entidades_conocidas",
        features=feats,
        validacion__esquema="kfold", validacion__n_folds=5, validacion__repeticiones=2,
        validacion__justificacion="Observaciones iid sin tiempo ni grupos: KFold repetido permite la t de Nadeau-Bengio.",
        holdout__estrategia="aleatorio", holdout__fraccion=0.2,
        metrica__principal="rmse", metrica__secundarias=["mae", "r2"],
        comparacion__margen_practico=0.05, comparacion__n_boot=300,
        diagnostico__grupos_features={"lineales": ["x1", "x2"], "interaccion": ["x3", "x4"]},
    )


GENERADORES = {
    "forecasting": (forecasting, contrato_forecasting, "ventas.csv"),
    "classification": (classification, contrato_classification, "churn.csv"),
    "regression": (regression, contrato_regression, "tabla.csv"),
}


def crear_run(destino: Path | str, tipo: str, con_fuga: bool = True, **ajustes) -> Path:
    """Crea un run con datos + contrato completos (y decisions.md con una decisión inicial)."""
    gen, con, archivo = GENERADORES[tipo]
    run = Path(destino)
    (run / "data").mkdir(parents=True, exist_ok=True)
    (run / "experiments").mkdir(exist_ok=True)
    gen(con_fuga=con_fuga).to_csv(run / "data" / archivo, index=False)
    c = con(con_fuga=con_fuga)
    for ruta, v in ajustes.items():
        cur = c
        partes = ruta.split("__")
        for k in partes[:-1]:
            cur = cur.setdefault(k, {})
        cur[partes[-1]] = v
    (run / "contract.yaml").write_text(yaml.safe_dump(c, allow_unicode=True, sort_keys=False), encoding="utf-8")
    (run / "decisions.md").write_text(
        (TEMPLATES / "decisions.md").read_text(encoding="utf-8")
        + "\n## D-001 — Esquema de validación\nFecha: 2026-10-22\nDecisión: "
        + c["validacion"]["justificacion"] + "\nJustificación: ver contract.yaml (validacion.justificacion) y eda/profile.json.\n",
        encoding="utf-8")
    return run


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tipo", choices=list(GENERADORES), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limpio", action="store_true")
    a = ap.parse_args()
    print(crear_run(a.out, a.tipo, con_fuga=not a.limpio))
