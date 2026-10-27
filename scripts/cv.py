"""
Splitters del harness y validación de splits.

Esquemas (campo `validacion.esquema` del contrato):

  temporal       Ventanas de validación consecutivas al final de la serie (origen móvil).
                 Fold k: validación = periodos [v_ini, v_fin]; entrenamiento = periodos
                 <= v_ini - gap - 1. `origen` = v_ini - 1 es el momento de predicción y
                 `gap` cubre el retraso con que se conoce el target (retraso_target).
                 El horizonte de cada fila de validación es `_t - origen`.
  grupos         GroupKFold (StratifiedGroupKFold en clasificación). Ninguna entidad está
                 a la vez en entrenamiento y validación.
  estratificado  (Repeated)StratifiedKFold. Solo clasificación sin dependencia temporal/grupal.
  kfold          (Repeated)KFold. Solo sin dependencia temporal/grupal.

Los splits se MATERIALIZAN en `splits.json` (ids de fila por fold) para que todos los
candidatos usen exactamente los mismos y el auditor pueda reconstruirlos y compararlos.

Uso directo (diagnóstico):
    python scripts/cv.py --run runs/<p>     # imprime resumen y valida los splits guardados
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import (GroupKFold, KFold, RepeatedKFold, RepeatedStratifiedKFold,
                                     StratifiedGroupKFold, StratifiedKFold)

from common import HarnessError, cli, read_json, sha256_json


def _fold(fid, rep, k, train_ids, val_ids, origen=None, gap=None):
    return {"id": fid, "repeticion": int(rep), "fold": int(k),
            "train": [str(i) for i in train_ids], "val": [str(i) for i in val_ids],
            "origen": None if origen is None else int(origen), "gap": None if gap is None else int(gap)}


def split_temporal(dev: pd.DataFrame, id_col: str, n_folds: int, tam_val: int, gap: int,
                   min_train: int = 1) -> list[dict]:
    periodos = np.sort(dev["_t"].unique())
    t_max = int(periodos.max())
    folds = []
    for k in range(n_folds):
        v_fin = t_max - (n_folds - 1 - k) * tam_val
        v_ini = v_fin - tam_val + 1
        tr_fin = v_ini - gap - 1
        tr = dev.loc[dev["_t"] <= tr_fin, id_col]
        va = dev.loc[(dev["_t"] >= v_ini) & (dev["_t"] <= v_fin), id_col]
        n_per_tr = int((periodos <= tr_fin).sum())
        if n_per_tr < min_train or len(va) == 0:
            raise HarnessError(
                f"el split temporal no cabe en los datos: fold {k} tendría {n_per_tr} periodos de entrenamiento "
                f"(mínimo {min_train}) y {len(va)} filas de validación.",
                "Reduce validacion.n_folds o validacion.tamano_validacion, o usa más historia.")
        folds.append(_fold(f"r0f{k}", 0, k, tr, va, origen=v_ini - 1, gap=gap))
    return folds


def split_grupos(dev, id_col, target, n_folds, reps, seed, clasificacion):
    folds = []
    g = dev["_entidad"].astype(str).values
    n_g = len(np.unique(g))
    if n_g < n_folds:
        raise HarnessError(f"hay {n_g} entidades y se piden {n_folds} folds por grupos.",
                           "Reduce validacion.n_folds.")
    for r in range(reps):
        if clasificacion:
            sp = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed + r)
            it = sp.split(dev, dev[target], groups=g)
        else:
            sp = GroupKFold(n_splits=n_folds, shuffle=True, random_state=seed + r)
            it = sp.split(dev, groups=g)
        for k, (tr, va) in enumerate(it):
            folds.append(_fold(f"r{r}f{k}", r, k, dev[id_col].values[tr], dev[id_col].values[va]))
    return folds


def split_aleatorio(dev, id_col, target, n_folds, reps, seed, estratificado):
    folds = []
    if estratificado:
        sp = (RepeatedStratifiedKFold(n_splits=n_folds, n_repeats=reps, random_state=seed) if reps > 1
              else StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed))
        it = sp.split(dev, dev[target])
    else:
        sp = (RepeatedKFold(n_splits=n_folds, n_repeats=reps, random_state=seed) if reps > 1
              else KFold(n_splits=n_folds, shuffle=True, random_state=seed))
        it = sp.split(dev)
    for j, (tr, va) in enumerate(it):
        r, k = divmod(j, n_folds)
        folds.append(_fold(f"r{r}f{k}", r, k, dev[id_col].values[tr], dev[id_col].values[va]))
    return folds


def make_splits(dev: pd.DataFrame, c: dict) -> dict:
    """Materializa los folds según el contrato `c`. `dev` debe traer `_t`/`_entidad` si aplican."""
    v = c["validacion"]
    d = c["datos"]
    esquema = v["esquema"]
    n = int(v.get("n_folds", 5))
    reps = int(v.get("repeticiones", 1) or 1)
    seed = int(v.get("semilla_split", 42))
    id_col, target = d["id_fila"], d["target"]
    if esquema == "temporal":
        folds = split_temporal(dev, id_col, n, int(v["tamano_validacion"]), int(v.get("gap", 0) or 0),
                               int(v.get("min_periodos_train", 1) or 1))
    elif esquema == "grupos":
        folds = split_grupos(dev, id_col, target, n, reps, seed, c["problem_type"] == "classification")
    elif esquema in ("estratificado", "kfold"):
        folds = split_aleatorio(dev, id_col, target, n, reps, seed, esquema == "estratificado")
    else:
        raise HarnessError(f"esquema de validación desconocido: {esquema}",
                           "Usa temporal | grupos | estratificado | kfold.")
    out = {"esquema": esquema, "n_folds": n, "repeticiones": reps if esquema != "temporal" else 1,
           "semilla_split": seed, "folds": folds}
    out["hash"] = sha256_json(folds)
    return out


def validar_splits(dev: pd.DataFrame, splits: dict, c: dict) -> list[str]:
    """Devuelve violaciones COMPROBADAS de los splits respecto del contrato (lista vacía = ok)."""
    d, v = c["datos"], c["validacion"]
    id_col = d["id_fila"]
    ids = dev[id_col].astype(str)
    t = dict(zip(ids, dev["_t"])) if "_t" in dev else None
    ent = dict(zip(ids, dev["_entidad"].astype(str))) if "_entidad" in dev else None
    gen = c["prediccion"].get("generalizacion")
    probs = []
    if sha256_json(splits["folds"]) != splits.get("hash"):
        probs.append("el contenido de splits.json no coincide con su hash interno (fue modificado).")
    validos = set(ids)
    cubre: dict[int, set] = {}
    for f in splits["folds"]:
        tr, va = set(f["train"]), set(f["val"])
        if tr & va:
            probs.append(f"fold {f['id']}: {len(tr & va)} filas están a la vez en entrenamiento y validación.")
        fuera = (tr | va) - validos
        if fuera:
            probs.append(f"fold {f['id']}: {len(fuera)} ids no existen en desarrollo (¿holdout o datos cambiados?).")
        cubre.setdefault(f["repeticion"], set()).update(va)
        if splits["esquema"] == "temporal" and t is not None and tr and va:
            gap = int(v.get("gap", 0) or 0)
            tr_max = max(t[i] for i in tr if i in t)
            va_min = min(t[i] for i in va if i in t)
            if va_min - tr_max - 1 < gap:
                probs.append(f"fold {f['id']}: la separación entre entrenamiento (t<={tr_max}) y validación "
                             f"(t>={va_min}) es menor que el gap declarado ({gap}).")
        if (splits["esquema"] == "grupos" or gen == "entidades_nuevas") and ent is not None:
            comun = {ent[i] for i in tr if i in ent} & {ent[i] for i in va if i in ent}
            if comun:
                probs.append(f"fold {f['id']}: {len(comun)} entidades aparecen en entrenamiento y validación "
                             f"(prohibido por generalización a entidades nuevas).")
    if splits["esquema"] != "temporal":
        for r, s in cubre.items():
            if s != validos:
                probs.append(f"repetición {r}: la validación cubre {len(s)} de {len(validos)} filas.")
    return probs


def resumen(splits: dict) -> str:
    lines = [f"esquema={splits['esquema']} folds={len(splits['folds'])}"]
    for f in splits["folds"]:
        extra = f" origen={f['origen']} gap={f['gap']}" if f["origen"] is not None else ""
        lines.append(f"  {f['id']}: train={len(f['train'])} val={len(f['val'])}{extra}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    a = ap.parse_args()
    from contract import cargar_contrato, cargar_dev
    c = cargar_contrato(a.run)
    sp = read_json(f"{a.run}/splits.json")
    if sp is None:
        raise HarnessError("no existe splits.json.", "Corre `python scripts/contract.py validate --run ...`.")
    print(resumen(sp))
    probs = validar_splits(cargar_dev(a.run), sp, c)
    for p in probs:
        print("VIOLACIÓN:", p)
    return 1 if probs else 0


if __name__ == "__main__":
    cli(main)
