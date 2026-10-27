"""
Utilidades compartidas del ds-harness.

Contiene lo que todos los scripts necesitan y nada de lógica de fases:
  * `HarnessError` y `cli()`: todo error de gate dice QUÉ falta y CÓMO resolverlo, y sale
    con código distinto de cero.
  * Hashes (`sha256_file`, `sha256_json`), escritura atómica y bloqueo por archivo, para que
    dos ejecuciones simultáneas no corrompan `state.json` ni `experiments.jsonl`.
  * Lectura/escritura de tablas: parquet si hay motor (pyarrow/fastparquet); si no, pickle.
    Los datos de entrada pueden ser CSV o parquet.
  * Estado del estudio (`state.json`) y registro de versiones del entorno.

Garantía real: los hashes detectan modificaciones accidentales o inconsistentes. No impiden
que alguien con permisos completos modifique a la vez un artefacto y su hash.
"""
from __future__ import annotations

import contextlib
import hashlib
import importlib
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8")

HARNESS_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = HARNESS_ROOT / "scripts"

FASES = ["contrato", "eda", "baselines", "experimentos", "diagnostico", "revision", "holdout", "reporte"]


# --------------------------------------------------------------------------- errores

class HarnessError(Exception):
    """Error que bloquea el avance. `que` = qué falta; `como` = cómo resolverlo."""

    def __init__(self, que: str, como: str, codigo: int = 2):
        super().__init__(que)
        self.que = que
        self.como = como
        self.codigo = codigo

    def mensaje(self) -> str:
        return f"BLOQUEADO: {self.que}\nCómo resolverlo: {self.como}"


def cli(fn: Callable[[], int | None]) -> None:
    """Ejecuta `fn` como entrada de línea de comandos traduciendo HarnessError a exit code."""
    try:
        rc = fn()
    except HarnessError as e:
        print(e.mensaje(), file=sys.stderr)
        sys.exit(e.codigo)
    sys.exit(rc or 0)


# --------------------------------------------------------------------------- tiempo / hashes

def ahora() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path | str) -> str | None:
    p = Path(path)
    if not p.exists():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _json_default(o: Any):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (pd.Timestamp, datetime)):
        return o.isoformat()
    if isinstance(o, Path):
        return o.as_posix()
    return str(o)


def dumps(obj: Any, indent: int | None = 2) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=indent, sort_keys=False, default=_json_default)


def sha256_json(obj: Any) -> str:
    txt = json.dumps(obj, ensure_ascii=False, sort_keys=True, default=_json_default, separators=(",", ":"))
    return hashlib.sha256(txt.encode("utf-8")).hexdigest()


def sha256_text(txt: str) -> str:
    return hashlib.sha256(txt.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- escritura atómica y bloqueo

def atomic_write_text(path: Path | str, text: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f".{p.name}.{os.getpid()}.tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    for intento in range(20):  # en Windows os.replace falla si otro proceso tiene el archivo abierto
        try:
            os.replace(tmp, p)
            return
        except PermissionError:
            time.sleep(0.1 * (intento + 1))
    os.replace(tmp, p)


def atomic_write_json(path: Path | str, obj: Any) -> None:
    atomic_write_text(path, dumps(obj) + "\n")


def read_json(path: Path | str, default: Any = None) -> Any:
    p = Path(path)
    if not p.exists():
        return default
    return json.loads(p.read_text(encoding="utf-8"))


@contextlib.contextmanager
def file_lock(run: Path | str, nombre: str = ".lock", timeout: float = 60.0, stale: float = 900.0):
    """Bloqueo exclusivo por archivo (O_EXCL). Un lock con más de `stale` segundos se considera
    huérfano (proceso muerto) y se reemplaza."""
    lock = Path(run) / nombre
    lock.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    while True:
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"{os.getpid()} {ahora()}".encode())
            os.close(fd)
            break
        except FileExistsError:
            try:
                if time.time() - lock.stat().st_mtime > stale:
                    lock.unlink(missing_ok=True)
                    continue
            except FileNotFoundError:
                continue
            if time.time() - t0 > timeout:
                raise HarnessError(f"otro proceso tiene bloqueado el estudio ({lock}).",
                                   "Espera a que termine la otra ejecución. Si no hay ninguna, borra el archivo .lock.")
            time.sleep(0.1)
    try:
        yield
    finally:
        lock.unlink(missing_ok=True)


# --------------------------------------------------------------------------- JSONL (solo se agrega)

def read_jsonl(path: Path | str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    out = []
    for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise HarnessError(f"{p.name} línea {i} no es JSON válido ({e}).",
                                   "El archivo se escribe solo con experiment_log.py; restáuralo desde git o vuelve a correr los experimentos.")
    return out


def append_jsonl(run: Path | str, path: Path | str, rec: dict) -> None:
    """Agrega un registro de forma atómica (reescribe a un temporal y reemplaza) bajo bloqueo."""
    p = Path(path)
    with file_lock(run, ".jsonl.lock"):
        prev = p.read_text(encoding="utf-8") if p.exists() else ""
        if prev and not prev.endswith("\n"):
            prev += "\n"
        atomic_write_text(p, prev + json.dumps(rec, ensure_ascii=False, default=_json_default) + "\n")


# --------------------------------------------------------------------------- tablas

def _parquet_engine() -> str | None:
    for eng in ("pyarrow", "fastparquet"):
        try:
            importlib.import_module(eng)
            return eng
        except ImportError:
            pass
    return None


PARQUET_ENGINE = _parquet_engine()
TABLE_EXT = ".parquet" if PARQUET_ENGINE else ".pkl"


def write_table(df: pd.DataFrame, base: Path | str) -> Path:
    """Escribe `df` en `base` + extensión según el motor disponible. Devuelve la ruta final."""
    p = Path(base)
    p = p.with_name(p.name.split(".")[0] + TABLE_EXT)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f".{p.name}.{os.getpid()}.tmp")
    if PARQUET_ENGINE:
        df.to_parquet(tmp, index=False, engine=PARQUET_ENGINE)
    else:
        df.to_pickle(tmp)
    os.replace(tmp, p)
    return p


def find_table(base: Path | str) -> Path | None:
    p = Path(base)
    stem = p.name.split(".")[0]
    for ext in (".parquet", ".pkl"):
        q = p.with_name(stem + ext)
        if q.exists():
            return q
    return None


def read_table(path: Path | str) -> pd.DataFrame:
    p = Path(path)
    suf = p.suffix.lower()
    if suf == ".parquet":
        if not PARQUET_ENGINE:
            raise HarnessError(f"{p.name} es parquet y no hay motor parquet instalado.",
                               "Instala pyarrow (`pip install pyarrow`) o exporta los datos a CSV.")
        return pd.read_parquet(p, engine=PARQUET_ENGINE)
    if suf == ".pkl":
        return pd.read_pickle(p)
    if suf in (".csv", ".txt"):
        return pd.read_csv(p)
    raise HarnessError(f"formato no soportado: {p.name}", "Usa CSV o parquet.")


def require_table(base: Path | str, que: str, como: str) -> pd.DataFrame:
    p = find_table(base)
    if p is None:
        raise HarnessError(que, como)
    return read_table(p)


# --------------------------------------------------------------------------- entorno

PAQUETES = ["numpy", "pandas", "scipy", "sklearn", "statsmodels", "yaml", "matplotlib", "pyarrow", "xgboost", "lightgbm"]


def disponible(mod: str) -> bool:
    try:
        importlib.import_module(mod)
        return True
    except ImportError:
        return False


def env_info() -> dict:
    vers = {}
    for m in PAQUETES:
        try:
            vers[m] = getattr(importlib.import_module(m), "__version__", "?")
        except ImportError:
            vers[m] = None
    return {"python": sys.version.split()[0], "plataforma": platform.platform(), "paquetes": vers,
            "motor_parquet": PARQUET_ENGINE}


# --------------------------------------------------------------------------- estado

def run_path(run: Path | str) -> Path:
    p = Path(run).resolve()
    return p


def default_state() -> dict:
    return {
        "version_estudio": 1,
        "fases": {},
        "holdout": {"estado": "sin_sellar", "observaciones_previas": [], "eventos": []},
        "eventos": [],
    }


def load_state(run: Path | str) -> dict:
    st = read_json(Path(run) / "state.json")
    return st if st is not None else default_state()


def save_state(run: Path | str, st: dict) -> None:
    atomic_write_json(Path(run) / "state.json", st)


@contextlib.contextmanager
def state_tx(run: Path | str):
    """Transacción sobre state.json: bloquea, carga, deja modificar y guarda atómicamente."""
    with file_lock(run):
        st = load_state(run)
        yield st
        save_state(run, st)


def log_event(st: dict, tipo: str, **datos) -> None:
    st.setdefault("eventos", []).append({"fecha": ahora(), "tipo": tipo, **datos})


def hash_rels(run: Path | str, rels: Iterable[str]) -> dict[str, str | None]:
    r = Path(run)
    return {rel: sha256_file(r / rel) for rel in rels}


def rel_table(run: Path | str, base_rel: str) -> str | None:
    """Ruta relativa real de una tabla interna (resuelve la extensión .parquet/.pkl)."""
    p = find_table(Path(run) / base_rel)
    return p.relative_to(Path(run)).as_posix() if p else None
