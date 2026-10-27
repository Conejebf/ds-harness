import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests" / "fixtures"))


def correr(script: str, *args, run=None, check=None):
    """Corre scripts/<script> como lo haría el agente. Devuelve CompletedProcess.
    check=True exige exit 0; check=False exige exit distinto de 0."""
    cmd = [sys.executable, str(ROOT / "scripts" / script), *map(str, args)]
    if run is not None:
        cmd += ["--run", str(run)]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", cwd=ROOT)
    if check is True and r.returncode != 0:
        raise AssertionError(f"{script} {args} falló ({r.returncode}):\n{r.stdout}\n{r.stderr}")
    if check is False and r.returncode == 0:
        raise AssertionError(f"{script} {args} debía fallar y pasó:\n{r.stdout}")
    return r
