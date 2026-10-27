"""
PreToolUse del ds-harness: protege los artefactos que solo deben escribir los scripts.

Bloquea (exit 2; el motivo va a stderr y Claude lo ve):
  * Write/Edit/MultiEdit/NotebookEdit sobre artefactos generados por scripts (state.json,
    experiments.jsonl, splits.json, comparison.json, audit.json, review.json, preds/, ...).
  * Read/Grep/Glob y comandos de shell que mencionen `_sealed` (el holdout solo lo abre holdout.py).
  * Comandos de shell que mencionen un artefacto protegido junto con una operación de escritura
    (redirección, rm/mv/cp, Set-Content, sed -i, open(..., 'w'), ...).

Límite: es una barrera contra errores y atajos, no de seguridad. Un comando ofuscado puede
evadirla; los hashes de state.json hacen visible la manipulación, pero no la impiden.
"""
from __future__ import annotations

import json
import re
import sys

for _s in (sys.stdin, sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8")

PROTEGIDO = re.compile(
    r"(^|[\\/])(state\.json|experiments\.jsonl|splits\.json|snapshot\.json|holdout_manifest\.json|baselines\.json|"
    r"comparison\.json|leakage_report\.json|audit\.json|review\.json|holdout_freeze\.json|holdout_result\.json|"
    r"diagnostics\.json|dev\.parquet|dev\.pkl|holdout_preds\.parquet|holdout_preds\.pkl)$"
    r"|(^|[\\/])(_sealed|preds|_executed)([\\/]|$)", re.IGNORECASE)
NOMBRE_PROTEGIDO = re.compile(
    r"(state\.json|experiments\.jsonl|splits\.json|snapshot\.json|holdout_manifest\.json|baselines\.json|comparison\.json|"
    r"leakage_report\.json|audit\.json|review\.json|holdout_freeze\.json|holdout_result\.json|diagnostics\.json|"
    r"dev\.parquet|dev\.pkl|[\\/]preds[\\/]|_executed)", re.IGNORECASE)
ESCRITURA = re.compile(
    r"(>|\btee\b|\brm\b|\bdel\b|\bmv\b|\bcp\b|\bmove\b|\bcopy\b|Set-Content|Add-Content|Out-File|Remove-Item|Move-Item|"
    r"Copy-Item|Clear-Content|\bsed\s+-i|\btruncate\b|write_text|write_bytes|\.write\(|json\.dump\(|to_parquet|to_csv|"
    r"open\([^)]*['\"][wa])", re.IGNORECASE)

QUIEN = {
    "state.json": "los scripts (contract.py, gate.py, holdout.py, final_report.py)",
    "experiments.jsonl": "experiment_log.py y baselines.py",
    "review.json": "review.py submit (con el dictamen del revisor)",
    "audit.json": "audit.py",
}


def bloquear(motivo: str) -> None:
    sys.stderr.write(f"BLOQUEADO por el ds-harness (pre_guard): {motivo}\n")
    sys.exit(2)


def main() -> None:
    try:
        data = json.load(sys.stdin)
    except json.JSONDecodeError:
        return
    tool = data.get("tool_name", "")
    ti = data.get("tool_input", {}) or {}

    if tool in ("Write", "Edit", "MultiEdit", "NotebookEdit"):
        path = (ti.get("file_path") or ti.get("notebook_path") or "").replace("\\", "/")
        if PROTEGIDO.search(path):
            nombre = path.rsplit("/", 1)[-1]
            bloquear(f"'{path}' es un artefacto que solo escribe {QUIEN.get(nombre, 'su script del harness')}. "
                     "Corre el script correspondiente; no edites el artefacto a mano.")
        return

    if tool in ("Read", "Grep", "Glob"):
        blob = " ".join(str(ti.get(k, "")) for k in ("file_path", "path", "pattern", "glob"))
        if "_sealed" in blob:
            bloquear("el holdout sellado (_sealed/) solo lo abre `scripts/holdout.py evaluate`, una vez, al final.")
        return

    if tool in ("Bash", "PowerShell"):
        cmd = ti.get("command", "")
        if "_sealed" in cmd:
            bloquear("el comando menciona _sealed/: el holdout solo lo abre `scripts/holdout.py evaluate`.")
        if NOMBRE_PROTEGIDO.search(cmd) and ESCRITURA.search(cmd):
            bloquear("el comando parece escribir, mover o borrar un artefacto protegido del harness. "
                     "Usa el script correspondiente (contract.py, experiment_log.py, compare_models.py, review.py, holdout.py...).")
        return


if __name__ == "__main__":
    main()
