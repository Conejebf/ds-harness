"""
Fase 1 — Perfil factual de los datos (parte automática del EDA).

Uso:
    python scripts/profile_data.py --run runs/<p>
    python scripts/profile_data.py --run runs/<p> --exploratorio [--datos ruta.csv]

Modo normal (requiere contrato validado): perfila SOLO los datos de desarrollo (el holdout está
sellado) y escribe:
    eda/profile.json      hechos estructurados (base de todo lo demás)
    eda/figuras/*.png
    eda_report.md         sección de hechos generada + bloque de leakage (lo llena
                          leakage_check.py) + sección `## Interpretación` que escribe el agente.
                          Si eda_report.md ya existe, se conservan la interpretación y el bloque de leakage.

Modo exploratorio (permitido con contrato incompleto): perfila el snapshot crudo en
eda/exploratorio/ y NO cuenta para ningún gate. Sirve para decidir el contrato, no para evaluar.

Hechos que calcula: esquema, faltantes, duplicados, distribución del target / balance de clases,
cardinalidades, outliers (regla 3xIQR), cobertura temporal y huecos por entidad, tendencia y
estacionalidad del target agregado, drift entre la primera y la segunda mitad del periodo
(KS en numéricas, PSI en categóricas) y correlaciones de Spearman con el target.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from common import (HarnessError, atomic_write_json, atomic_write_text, cli, find_table, read_json, read_table,
                    sha256_file)

MARCA_HECHOS = ("<!-- HECHOS:INICIO (generado por profile_data.py; no editar) -->", "<!-- HECHOS:FIN -->")
MARCA_LEAK = ("<!-- LEAKAGE:INICIO (generado por leakage_check.py; no editar) -->", "<!-- LEAKAGE:FIN -->")
INTERP = "## Interpretación"
PENDIENTE = "<!-- PENDIENTE: el agente escribe aquí su interpretación, citando artefactos (eda/profile.json, eda/figuras/..., leakage_report.json) y las preguntas abiertas. -->"


def _psi(a: pd.Series, b: pd.Series) -> float:
    cats = set(a.dropna().unique()) | set(b.dropna().unique())
    pa = a.value_counts(normalize=True)
    pb = b.value_counts(normalize=True)
    s = 0.0
    for k in cats:
        x, y = max(pa.get(k, 0), 1e-4), max(pb.get(k, 0), 1e-4)
        s += (x - y) * np.log(x / y)
    return float(s)


def perfilar(df: pd.DataFrame, c: dict | None, figdir: Path | None) -> dict:
    from scipy import stats
    d = (c or {}).get("datos", {}) or {}
    target, idc, tcol, ent = d.get("target"), d.get("id_fila"), d.get("tiempo"), d.get("entidad")
    pt = (c or {}).get("problem_type")
    cols = [x for x in df.columns if not x.startswith("_")]
    P: dict = {"filas": int(len(df)), "columnas": len(cols), "esquema": {k: str(df[k].dtype) for k in cols}}
    P["faltantes"] = {k: {"n": int(df[k].isna().sum()), "pct": round(float(df[k].isna().mean()) * 100, 2)}
                      for k in cols if df[k].isna().any()}
    contenido = [x for x in cols if x != idc]
    P["duplicados"] = {"filas_contenido_identico": int(df.duplicated(subset=contenido).sum()),
                       "ids_repetidos": int(df[idc].duplicated().sum()) if idc in df else None}
    P["cardinalidad"] = {k: int(df[k].nunique()) for k in cols}
    num = [k for k in cols if pd.api.types.is_numeric_dtype(df[k]) and k not in (idc,)]
    cat = [k for k in cols if k not in num and k not in (idc, tcol)]
    P["alta_cardinalidad"] = [k for k in cat if df[k].nunique() > 0.5 * len(df)]
    out = {}
    for k in num:
        s = df[k].dropna()
        if len(s) < 4:
            continue
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        n = int(((s < q1 - 3 * iqr) | (s > q3 + 3 * iqr)).sum()) if iqr > 0 else 0
        if n:
            out[k] = n
    P["outliers_3iqr"] = out

    if target in df:
        y = df[target]
        if pt == "classification":
            vc = y.value_counts()
            P["target"] = {"tipo": "clases", "conteos": {str(k): int(v) for k, v in vc.items()},
                           "proporcion_minoritaria": round(float(vc.min() / vc.sum()), 4)}
        else:
            P["target"] = {"tipo": "continuo", **{k: float(v) for k, v in y.describe().items()},
                           "asimetria": float(stats.skew(y)), "pct_ceros": round(float((y == 0).mean()) * 100, 2)}

    # tiempo y entidades
    if "_t" in df:
        P["tiempo"] = {"columna": tcol, "periodos": int(df["_t"].nunique()), "min": str(df[tcol].min()), "max": str(df[tcol].max())}
        if "_entidad" in df:
            g = df.groupby("_entidad")["_t"].agg(["min", "max", "count"])
            huecos = int(((g["max"] - g["min"] + 1) - g["count"]).clip(lower=0).sum())
            P["tiempo"]["huecos_en_grilla_por_entidad"] = huecos
    if "_entidad" in df:
        fe = df.groupby("_entidad").size()
        P["entidades"] = {"columna": ent, "n": int(len(fe)), "filas_min": int(fe.min()), "filas_mediana": float(fe.median()),
                          "filas_max": int(fe.max())}

    # tendencia y estacionalidad del target agregado
    if "_t" in df and target in df and pt != "classification":
        serie = df.groupby("_t")[target].mean()
        import statsmodels.api as sm
        X = sm.add_constant(np.arange(len(serie)))
        fit = sm.OLS(serie.values, X).fit()
        lo, hi = fit.conf_int()[1]
        tend = {"pendiente_por_periodo": float(fit.params[1]), "ic95": [float(lo), float(hi)],
                "significativa": bool(lo > 0 or hi < 0)}
        est = (c or {}).get("prediccion", {}).get("estacionalidad")
        estac = None
        if est and len(serie) >= 3 * int(est):
            from statsmodels.tsa.seasonal import seasonal_decompose
            from statsmodels.tsa.stattools import acf
            dec = seasonal_decompose(serie.values, period=int(est), model="additive", extrapolate_trend="freq")
            resid_var = np.var(dec.resid)
            fuerza = max(0.0, 1 - resid_var / np.var(dec.resid + dec.seasonal))
            estac = {"periodo": int(est), "fuerza": round(float(fuerza), 3),
                     "acf_en_periodo": round(float(acf(serie.values, nlags=int(est))[int(est)]), 3)}
        P["serie_target"] = {"tendencia": tend, "estacionalidad": estac}

    # drift entre mitades
    if "_t" in df:
        corte = df["_t"].quantile(0.5)
        a, b = df[df["_t"] <= corte], df[df["_t"] > corte]
        dr = {}
        for k in num:
            if k in (tcol,):
                continue
            sa, sb = a[k].dropna(), b[k].dropna()
            if len(sa) > 10 and len(sb) > 10:
                ks = stats.ks_2samp(sa, sb)
                dr[k] = {"ks": round(float(ks.statistic), 4), "p": float(ks.pvalue)}
        for k in cat:
            dr[k] = {"psi": round(_psi(a[k], b[k]), 4)}
        P["drift_mitades"] = {"criterio": "primera vs segunda mitad del periodo de desarrollo", "por_columna": dr,
                              "marcadas": [k for k, v in dr.items() if v.get("ks", 0) > 0.2 or v.get("psi", 0) > 0.2]}
    else:
        P["drift_mitades"] = {"criterio": "no aplica: sin columna de tiempo", "por_columna": {}, "marcadas": []}

    # correlaciones con el target
    corr = {}
    if target in df:
        yv = df[target]
        if pt == "classification":
            yv = yv.astype("category").cat.codes
        for k in num:
            if k == target:
                continue
            m = df[k].notna()
            if m.sum() > 10 and df.loc[m, k].nunique() > 1:
                corr[k] = round(float(stats.spearmanr(df.loc[m, k], yv[m]).statistic), 4)
    umbral = float(((c or {}).get("leakage") or {}).get("umbral_correlacion", 0.95))
    P["correlacion_spearman_target"] = dict(sorted(corr.items(), key=lambda kv: -abs(kv[1])))
    P["correlaciones_sospechosas"] = [k for k, v in corr.items() if abs(v) >= umbral]

    if figdir is not None:
        P["figuras"] = _figuras(df, P, target, pt, figdir)
    return P


def _figuras(df, P, target, pt, figdir: Path) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figdir.mkdir(parents=True, exist_ok=True)
    out = []

    def guardar(nombre):
        p = figdir / nombre
        plt.tight_layout()
        plt.savefig(p, dpi=90)
        plt.close()
        out.append(p.name)

    if target in df:
        plt.figure(figsize=(5, 3))
        if pt == "classification":
            df[target].value_counts().plot.bar()
        else:
            plt.hist(df[target], bins=40)
        plt.title(f"Distribución de {target}")
        guardar("target.png")
    if "_t" in df and target in df and pt != "classification":
        plt.figure(figsize=(7, 3))
        df.groupby("_t")[target].mean().plot()
        plt.title("Target promedio por periodo")
        guardar("serie_target.png")
    cr = P.get("correlacion_spearman_target", {})
    if cr:
        plt.figure(figsize=(5, 0.3 * len(cr) + 1))
        s = pd.Series(cr).sort_values()
        s.plot.barh()
        plt.title("Spearman con el target")
        guardar("correlaciones.png")
    return out


def _hechos_md(P: dict, run_rel: str) -> str:
    L = [f"- Filas (desarrollo): {P['filas']}; columnas: {P['columnas']}.",
         f"- Duplicados: {P['duplicados']}.",
         f"- Faltantes: {P['faltantes'] or 'ninguno'}.",
         f"- Outliers (3xIQR): {P['outliers_3iqr'] or 'ninguno'}.",
         f"- Alta cardinalidad: {P['alta_cardinalidad'] or 'ninguna'}."]
    if "target" in P:
        L.append(f"- Target: {P['target']}.")
    for k in ("tiempo", "entidades", "serie_target"):
        if k in P:
            L.append(f"- {k}: {P[k]}.")
    L.append(f"- Drift entre mitades ({P['drift_mitades']['criterio']}): marcadas {P['drift_mitades']['marcadas'] or 'ninguna'}.")
    L.append(f"- Correlaciones de Spearman con el target ≥ umbral: {P['correlaciones_sospechosas'] or 'ninguna'}.")
    for f in P.get("figuras", []):
        L.append(f"- Figura: `eda/figuras/{f}`")
    L.append(f"- Detalle completo: `eda/profile.json`.")
    return "\n".join(L)


def reemplazar_bloque(texto: str, marcas: tuple[str, str], contenido: str) -> str:
    ini, fin = marcas
    bloque = f"{ini}\n{contenido}\n{fin}"
    if ini in texto and fin in texto:
        return re.sub(re.escape(ini) + r".*?" + re.escape(fin), lambda _: bloque, texto, flags=re.S)
    return texto.rstrip() + "\n\n" + bloque + "\n"


def escribir_reporte(run: Path, P: dict) -> None:
    p = run / "eda_report.md"
    if p.exists():
        txt = p.read_text(encoding="utf-8")
    else:
        txt = ("# Reporte EDA\n\n## Hechos\n\n" + MARCA_HECHOS[0] + "\n" + MARCA_HECHOS[1] + "\n\n## Leakage\n\n"
               + MARCA_LEAK[0] + "\nVeredicto de leakage: PENDIENTE (corre leakage_check.py)\n" + MARCA_LEAK[1]
               + f"\n\n{INTERP}\n\n{PENDIENTE}\n")
    txt = reemplazar_bloque(txt, MARCA_HECHOS, _hechos_md(P, run.name))
    atomic_write_text(p, txt)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--exploratorio", action="store_true")
    ap.add_argument("--datos", help="(exploratorio) ruta del snapshot; por defecto datos.ruta del contrato")
    a = ap.parse_args()
    run = Path(a.run)
    if a.exploratorio:
        import yaml
        c = yaml.safe_load((run / "contract.yaml").read_text(encoding="utf-8")) if (run / "contract.yaml").exists() else {}
        ruta = a.datos or ((c.get("datos") or {}).get("ruta"))
        if not ruta or ruta == "POR_DEFINIR":
            raise HarnessError("no sé qué datos perfilar.", "Pasa --datos ruta.csv o define datos.ruta en contract.yaml.")
        src = Path(ruta) if Path(ruta).is_absolute() else run / ruta
        df = read_table(src)
        P = perfilar(df, {"datos": {"id_fila": (c.get("datos") or {}).get("id_fila")}}, run / "eda" / "exploratorio" / "figuras")
        P["advertencia"] = "Perfil EXPLORATORIO del snapshot crudo (incluye lo que será holdout). No cuenta para ningún gate."
        atomic_write_json(run / "eda" / "exploratorio" / "profile.json", P)
        print(f"OK: perfil exploratorio en {run / 'eda' / 'exploratorio'} (no aprueba ninguna fase).")
        return 0
    from contract import cargar_contrato, cargar_dev
    from gate import requerir
    requerir(run, "eda")
    c = cargar_contrato(run)
    df = cargar_dev(run)
    P = perfilar(df, c, run / "eda" / "figuras")
    P["dev_sha256"] = sha256_file(find_table(run / "dev"))
    P["manifest_holdout_duplicados"] = (read_json(run / "holdout_manifest.json") or {}).get("duplicados")
    atomic_write_json(run / "eda" / "profile.json", P)
    escribir_reporte(run, P)
    print(f"OK: perfil en eda/profile.json; hechos actualizados en eda_report.md. "
          f"Correlaciones sospechosas: {P['correlaciones_sospechosas'] or 'ninguna'}. Siguiente: leakage_check.py.")
    return 0


if __name__ == "__main__":
    cli(main)
