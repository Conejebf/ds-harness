"""
Capa LLM (opcional, cuesta tokens): ejecuta al revisor-adversarial REAL con `claude -p` sobre un
estudio en el que una fuga del futuro se "justificó" y llegó al modelo elegido, sin linaje
(la auditoría determinista solo puede marcarla como no_verificable + señal importante).

Se espera que el revisor la escale a hallazgo bloqueante. Registra entrada, salida y versión del
modelo en tests/llm_logs/. No reemplaza las garantías deterministas de test_audit_review.py.

    python -m pytest -m llm tests/test_llm_reviewer.py
"""
import json
import re
import shutil
import subprocess
import time
from pathlib import Path

import pytest

from conftest import ROOT, correr
from helpers import leer

pytestmark = pytest.mark.llm


def test_revisor_llm_encuentra_fuga_escapada(tmp_path):
    exe = shutil.which("claude")
    if not exe:
        pytest.skip("no está instalado el CLI `claude`")
    from test_audit_review import _escapar_fuga
    run = _escapar_fuga(tmp_path, con_linaje=False)
    correr("audit.py", run=run)
    correr("review.py", "prepare", run=run, check=True)
    agente = (ROOT / ".claude" / "agents" / "revisor-adversarial.md").read_text(encoding="utf-8").split("---", 2)[2]
    prompt = (f"Run: {run.as_posix()}\nBrief: {(run / 'reviews' / 'ronda_1' / 'brief.md').as_posix()}\n"
              "Haz la revisión adversarial completa de la ronda 1 y envía tu dictamen con review.py submit.")
    t0 = time.time()
    r = subprocess.run([exe, "-p", prompt, "--output-format", "json", "--append-system-prompt", agente,
                        "--allowedTools", "Read,Grep,Glob,Bash,Write"],
                       cwd=ROOT, capture_output=True, text=True, encoding="utf-8", timeout=3600)
    try:
        salida = json.loads(r.stdout)
    except json.JSONDecodeError:
        salida = {"crudo": r.stdout}
    logs = ROOT / "tests" / "llm_logs"
    logs.mkdir(exist_ok=True)
    (logs / f"revisor_{time.strftime('%Y%m%d_%H%M%S')}.json").write_text(json.dumps({
        "prompt": prompt, "system_append": agente, "returncode": r.returncode, "stderr": r.stderr[-5000:],
        "modelos": list((salida.get("modelUsage") or {}).keys()), "duracion_s": round(time.time() - t0, 1),
        "salida": salida, "review_json": leer(run, "review.json") if (run / "review.json").exists() else None,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    if not (run / "review.json").exists() and re.search(r"(Failed to authenticate|not logged in|OAuth|/login)", r.stdout + r.stderr, re.I):
        pytest.skip("el CLI `claude` no está autenticado (vuelve a iniciar sesión con `claude` → /login); no es un fallo del revisor")
    assert (run / "review.json").exists(), f"el revisor no envió dictamen:\n{r.stdout[-3000:]}\n{r.stderr[-2000:]}"
    R = leer(run, "review.json")
    del_revisor = [v for v in R["hallazgos"].values() if v["origen"] == "revisor"]
    sobre_fuga = [v for v in del_revisor if "indice_demanda" in (v["evidencia"] + (v.get("artefacto") or "") + (v.get("accion_requerida") or ""))]
    assert sobre_fuga, f"el revisor no reportó indice_demanda: {del_revisor}"
    assert any(v["severidad"] == "bloqueante" and v["estado"] == "abierto" for v in sobre_fuga), sobre_fuga
    assert "indice_demanda" in json.dumps(R, ensure_ascii=False)
    assert leer(run, "state.json")["holdout"]["estado"] == "sellado"   # no abrió el holdout
