"""Hook PreToolUse: protege artefactos generados y el holdout sellado."""
import json
import subprocess
import sys

import pytest

from conftest import ROOT

HOOK = ROOT / ".claude" / "hooks" / "pre_guard.py"


def hook(tool, **ti):
    r = subprocess.run([sys.executable, str(HOOK)], input=json.dumps({"tool_name": tool, "tool_input": ti}),
                       capture_output=True, text=True, encoding="utf-8")
    return r.returncode, r.stderr


@pytest.mark.parametrize("path", ["runs/x/state.json", "runs/x/experiments.jsonl", "runs\\x\\review.json", "runs/x/audit.json",
                                  "runs/x/_sealed/holdout.parquet", "runs/x/preds/E001.parquet", "runs/x/comparison.json",
                                  "runs/x/experiments/_executed/E001.py", "runs/x/diagnostics/diagnostics.json"])
def test_bloquea_edicion_de_artefactos(path):
    rc, err = hook("Write", file_path=path, content="{}")
    assert rc == 2 and "BLOQUEADO" in err
    assert hook("Edit", file_path=path, old_string="a", new_string="b")[0] == 2


@pytest.mark.parametrize("path", ["runs/x/contract.yaml", "runs/x/decisions.md", "runs/x/experiments/E003.py",
                                  "runs/x/eda_report.md", "runs/x/leakage_resolutions.yaml",
                                  "runs/x/reviews/ronda_1/review_draft.json", "runs/x/diagnostics/declaracion.md"])
def test_permite_archivos_del_agente(path):
    assert hook("Write", file_path=path, content="x")[0] == 0


def test_bloquea_lectura_del_holdout():
    assert hook("Read", file_path="runs/x/_sealed/holdout.parquet")[0] == 2
    assert hook("Glob", pattern="runs/**/_sealed/*")[0] == 2
    assert hook("Grep", pattern="ventas", path="runs/x/_sealed")[0] == 2
    assert hook("Read", file_path="runs/x/holdout_manifest.json")[0] == 0


@pytest.mark.parametrize("cmd", [
    "python -c \"import pandas as pd; print(pd.read_parquet('runs/x/_sealed/holdout.parquet'))\"",
    "echo '{}' > runs/x/state.json",
    "rm runs/x/experiments.jsonl",
    "Set-Content runs/x/review.json '{}'",
    "python -c \"open('runs/x/audit.json','w').write('{}')\"",
    "sed -i 's/abierto/resuelto/' runs/x/review.json",
])
def test_bloquea_comandos(cmd):
    assert hook("Bash", command=cmd)[0] == 2


@pytest.mark.parametrize("cmd", [
    "python scripts/experiment_log.py run E003 --run runs/x",
    "python scripts/holdout.py evaluate --run runs/x",
    "cat runs/x/state.json",
    "python scripts/audit.py --run runs/x > runs/x/audit_log.txt",
])
def test_permite_comandos_normales(cmd):
    assert hook("Bash", command=cmd)[0] == 0
