"""
Gates de fase del ds-harness.

Cada fase aprobada queda en `state.json` con:
  * `hashes`: sha256 de sus artefactos al momento de aprobarse, y
  * `cadena`: hash de los hashes de TODAS las fases anteriores.

Una fase es válida solo si sus artefactos no cambiaron y la cadena coincide; así, si se
re-aprueba una fase anterior con otros artefactos, todas las posteriores quedan invalidadas
y hay que rehacerlas. Ningún script avanza si las fases previas no son válidas.

Fases: contrato → eda → baselines → experimentos → diagnostico → revision → holdout → reporte.
La fase `contrato` la aprueba `contract.py validate`; `holdout` la aprueba `holdout.py`;
`reporte` la aprueba `final_report.py`. Las demás se aprueban con `gate.py advance <fase>`.

Uso:
    python scripts/gate.py status  --run runs/<p>           # estado de todas las fases
    python scripts/gate.py check   <fase> --run runs/<p>    # ¿puede ARRANCAR la fase? (exit 0/2)
    python scripts/gate.py advance <fase> --run runs/<p>    # verifica criterios de cierre y la aprueba

Límite: los hashes detectan cambios accidentales o inconsistentes; no impiden que alguien con
permisos completos reescriba artefactos y estado a la vez.
"""
from __future__ import annotations

import argparse
import importlib
from pathlib import Path

from common import FASES, HarnessError, ahora, cli, hash_rels, load_state, log_event, sha256_json, state_tx

# Qué módulo define los criterios de cierre de cada fase: función `criterios(run) -> (problemas, rels)`.
MODULO_CRITERIOS = {
    "contrato": "contract",
    "eda": "leakage_check",
    "baselines": "baselines",
    "experimentos": "compare_models",
    "diagnostico": "diagnostics",
    "revision": "review",
    "holdout": "holdout",
    "reporte": "final_report",
}

COMO_FASE = {
    "contrato": "Completa contract.yaml y corre `python scripts/contract.py validate --run {run}`.",
    "eda": "Corre profile_data.py y leakage_check.py, escribe la sección de Interpretación de eda_report.md y luego `python scripts/gate.py advance eda --run {run}`.",
    "baselines": "Corre `python scripts/baselines.py --run {run}` y luego `python scripts/gate.py advance baselines --run {run}`.",
    "experimentos": "Registra experimentos con experiment_log.py, corre compare_models.py y luego `python scripts/gate.py advance experimentos --run {run}`.",
    "diagnostico": "Corre `python scripts/diagnostics.py --run {run}` y luego `python scripts/gate.py advance diagnostico --run {run}`.",
    "revision": "Corre audit.py, lanza al revisor-adversarial, envía su dictamen con review.py submit y luego `python scripts/gate.py advance revision --run {run}`.",
    "holdout": "Corre `python scripts/holdout.py evaluate --run {run}`.",
    "reporte": "Corre `python scripts/final_report.py --run {run}`.",
}


def _cadena(st: dict, fase: str) -> str:
    previas = FASES[: FASES.index(fase)]
    return sha256_json([st["fases"].get(p, {}).get("hashes") for p in previas])


def estado_fase(run, st: dict, fase: str) -> tuple[str, str]:
    """('valida'|'invalida'|'pendiente', motivo)."""
    rec = st.get("fases", {}).get(fase)
    if not rec:
        return "pendiente", "no se ha aprobado"
    if rec.get("cadena") != _cadena(st, fase):
        return "invalida", "una fase anterior se re-aprobó con otros artefactos después de aprobar esta"
    actuales = hash_rels(run, rec["hashes"].keys())
    for rel, h in rec["hashes"].items():
        if actuales[rel] != h:
            que = "ya no existe" if actuales[rel] is None else "cambió"
            return "invalida", f"el artefacto {rel} {que} desde que se aprobó la fase"
    return "valida", ""


def requerir(run, fase: str) -> None:
    """Lanza HarnessError si alguna fase anterior a `fase` no está aprobada y válida."""
    st = load_state(run)
    for p in FASES[: FASES.index(fase)]:
        est, motivo = estado_fase(run, st, p)
        if est != "valida":
            raise HarnessError(f"no puede arrancar la fase '{fase}': la fase '{p}' está {est} ({motivo}).",
                               COMO_FASE[p].format(run=run))


def criterios(run, fase: str) -> tuple[list[tuple[str, str]], list[str]]:
    mod = importlib.import_module(MODULO_CRITERIOS[fase])
    return mod.criterios(run)


def registrar(run, fase: str, rels: list[str], **extra) -> None:
    """Aprueba `fase` guardando hashes de `rels`. Si cambian sus hashes, las fases posteriores
    quedan invalidadas por la cadena (se conservan en el historial de eventos)."""
    with state_tx(run) as st:
        hashes = hash_rels(run, rels)
        faltan = [r for r, h in hashes.items() if h is None]
        if faltan:
            raise HarnessError(f"no se puede aprobar '{fase}': faltan artefactos {faltan}.", COMO_FASE[fase].format(run=run))
        previo = st["fases"].get(fase, {}).get("hashes")
        st["fases"][fase] = {"estado": "aprobada", "fecha": ahora(), "hashes": hashes,
                             "cadena": _cadena(st, fase), **extra}
        if previo is not None and previo != hashes:
            log_event(st, "fase_reaprobada", fase=fase, invalida=FASES[FASES.index(fase) + 1:])
        log_event(st, "fase_aprobada", fase=fase)


def avanzar(run, fase: str) -> None:
    requerir(run, fase)
    probs, rels = criterios(run, fase)
    if probs:
        que = "\n".join(f"  - {q}" for q, _ in probs)
        como = "\n".join(f"  - {c}" for _, c in probs)
        raise HarnessError(f"la fase '{fase}' no cumple sus criterios de cierre:\n{que}", "\n" + como)
    registrar(run, fase, rels)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("accion", choices=["status", "check", "advance"])
    ap.add_argument("fase", nargs="?", choices=FASES)
    ap.add_argument("--run", required=True)
    a = ap.parse_args()
    run = str(Path(a.run))
    if a.accion == "status":
        st = load_state(run)
        print(f"Estudio {run} — versión {st.get('version_estudio')} — holdout: {st['holdout']['estado']}")
        rc = 0
        for p in FASES:
            est, motivo = estado_fase(run, st, p)
            print(f"  {p:<13} {est:<10} {motivo}")
        return rc
    if not a.fase:
        raise HarnessError("falta la fase.", f"Indica una de: {', '.join(FASES)}.")
    if a.accion == "check":
        requerir(run, a.fase)
        print(f"OK: la fase '{a.fase}' puede arrancar.")
        return 0
    if a.fase in ("contrato", "holdout", "reporte"):
        raise HarnessError(f"la fase '{a.fase}' no se aprueba con gate.py.", COMO_FASE[a.fase].format(run=run))
    avanzar(run, a.fase)
    print(f"OK: fase '{a.fase}' aprobada.")
    return 0


if __name__ == "__main__":
    cli(main)
