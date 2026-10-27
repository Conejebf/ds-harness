"""
Fase 5 — Revisión adversarial: rondas, dictamen estructurado y bloqueo ante hallazgos abiertos.

Uso:
    python scripts/review.py prepare --run runs/<p>
        Exige audit.json vigente. Crea reviews/ronda_N/brief.md con lo que el revisor debe ver
        (contrato, artefactos, código, decisiones, informes; NO la conversación del experimentador).
    python scripts/review.py submit --run runs/<p> --file reviews/ronda_N/review_draft.json
        Valida el dictamen contra el esquema, lo combina con las fallas de audit.json y escribe
        review.json (acumulado, con historial) y review.md (generado desde el JSON).

Esquema de review_draft.json:
    {"ronda": N, "revisor": "revisor-adversarial", "modelo": "<id del modelo>",
     "resumen": "...",
     "verificaciones": [{"nombre": "...", "resultado": "ok|falla|no_verificable", "detalle": "..."}],
     "hallazgos": [{"id": "R-001", "severidad": "bloqueante|importante|menor",
                    "estado": "abierto|resuelto|aceptado", "evidencia": "...", "artefacto": "...",
                    "accion_requerida": "...", "resolucion": "(obligatoria si resuelto/aceptado)"}]}

Reglas:
  * El revisor debe pronunciarse sobre TODOS los hallazgos abiertos de rondas anteriores.
  * Un bloqueante no se puede `aceptar`; solo `resolver`, y solo en una ronda posterior en la que
    los artefactos hayan cambiado (el experimentador aporta la corrección y se vuelve a revisar).
  * Las fallas de audit.py entran solas como hallazgos A-*; se cierran cuando audit.py deja de detectarlas.
  * El gate falla si hay bloqueantes abiertos o si la última ronda revisó una versión anterior
    de los artefactos.

Límite: el revisor corre con Bash, así que técnicamente no es de solo lectura; los hooks bloquean
la edición directa de review.json/audit.json/state.json, pero quien controle la sesión podría
escribir un dictamen falso. El valor de la revisión depende de que la haga un agente con contexto limpio.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

from common import HarnessError, ahora, atomic_write_json, atomic_write_text, cli, read_json, sha256_json

SEVERIDADES = ("bloqueante", "importante", "menor")
ESTADOS = ("abierto", "resuelto", "aceptado")


def _rondas(run) -> list[int]:
    d = Path(run) / "reviews"
    return sorted(int(p.name.split("_")[1]) for p in d.glob("ronda_*") if p.is_dir()) if d.exists() else []


def prepare(run) -> Path:
    from audit import conjunto_revisado, vigente
    from gate import requerir
    run = Path(run)
    requerir(run, "revision")
    ok, motivo = vigente(run)
    if not ok:
        raise HarnessError(f"audit.json no está vigente ({motivo}).", f"Corre `python scripts/audit.py --run {run}` primero.")
    R = read_json(run / "review.json") or {"rondas": [], "hallazgos": {}}
    hechas = {r["ronda"] for r in R["rondas"]}
    pend = [n for n in _rondas(run) if n not in hechas]
    n = pend[0] if pend else (max(_rondas(run) or [0]) + 1)
    d = run / "reviews" / f"ronda_{n}"
    d.mkdir(parents=True, exist_ok=True)
    abiertos = {k: v for k, v in R["hallazgos"].items() if v["estado"] == "abierto"}
    brief = f"""# Revisión adversarial — ronda {n}

Run: `{run}`. Tu trabajo es intentar ROMPER este estudio. No tienes (ni debes buscar) la conversación
del experimentador: solo artefactos y código.

## Qué leer
- contract.yaml, snapshot.json, holdout_manifest.json (NO abras _sealed/)
- eda_report.md, eda/profile.json, leakage_report.json, leakage_resolutions.yaml (si existe)
- splits.json, baselines.json, experiments.jsonl, experiments/_executed/*.py, preds/
- comparison.json / comparison.md, diagnostics/diagnostics.json, decisions.md
- audit.json (auditoría determinista ya corrida: {read_json(run / 'audit.json')['resumen']})

## Qué intentar (como mínimo)
1. Buscar leakage: features que no existirían al predecir, derivadas del target, justificaciones débiles en leakage_resolutions.yaml.
2. Validar que el split sea el declarado (re-construirlo tú: `python scripts/cv.py --run {run}`).
3. RE-EJECUTAR métricas desde cero (`python scripts/audit.py --run {run} --alcance todos`, o tu propio script de lectura) y compararlas con lo reportado.
4. Sobreajuste al CV por número de experimentos; ¿el "ganador" lo es por ruido? ¿Holm/IC se interpretaron bien?
5. Verificar que ninguna decisión se tomó mirando el holdout.
6. Revisar que decisions.md justifique cada decisión no trivial y que el cierre de la exploración (si lo hubo) se sostenga.

## Hallazgos abiertos de rondas anteriores (debes pronunciarte sobre cada uno)
{chr(10).join(f"- {k} [{v['severidad']}]: {v['evidencia']}" for k, v in abiertos.items()) or "- ninguno"}

## Entrega
Escribe `reviews/ronda_{n}/review_draft.json` con el esquema de `scripts/review.py` y corre:
`python scripts/review.py submit --run {run} --file reviews/ronda_{n}/review_draft.json`
"""
    atomic_write_text(d / "brief.md", brief)
    atomic_write_json(d / "artefactos.json", conjunto_revisado(run))
    return d


def _validar_draft(D: dict, n: int) -> list[str]:
    E = []
    if D.get("ronda") != n:
        E.append(f"`ronda` debe ser {n}")
    for k in ("revisor", "modelo", "resumen"):
        if not str(D.get(k, "")).strip():
            E.append(f"falta `{k}`")
    if not D.get("verificaciones"):
        E.append("`verificaciones` vacío: documenta qué re-ejecutaste y comprobaste")
    for v in D.get("verificaciones", []):
        if v.get("resultado") not in ("ok", "falla", "no_verificable") or not v.get("nombre"):
            E.append(f"verificación inválida: {v}")
    ids = set()
    for h in D.get("hallazgos", []):
        hid = h.get("id", "")
        if not re.fullmatch(r"R-\d{3}", hid) and not hid.startswith("A-"):
            E.append(f"id inválido {hid!r} (usa R-001, R-002...)")
        if hid in ids:
            E.append(f"id repetido {hid}")
        ids.add(hid)
        if h.get("severidad") not in SEVERIDADES:
            E.append(f"{hid}: severidad inválida")
        if h.get("estado") not in ESTADOS:
            E.append(f"{hid}: estado inválido")
        for k in ("evidencia", "artefacto", "accion_requerida"):
            if not str(h.get(k, "")).strip():
                E.append(f"{hid}: falta `{k}`")
        if h.get("estado") in ("resuelto", "aceptado") and len(str(h.get("resolucion", "")).strip()) < 20:
            E.append(f"{hid}: `resolucion` obligatoria (>= 20 caracteres) para estado {h.get('estado')}")
        if h.get("estado") == "aceptado" and h.get("severidad") == "bloqueante":
            E.append(f"{hid}: un bloqueante no se puede aceptar; solo resolver")
    return E


def submit(run, archivo) -> dict:
    from audit import conjunto_revisado, vigente
    from gate import requerir
    run = Path(run)
    requerir(run, "revision")
    p = Path(archivo) if Path(archivo).is_absolute() else (run / archivo if (run / archivo).exists() else Path(archivo))
    D = read_json(p)
    if D is None:
        raise HarnessError(f"no existe {archivo}.", "El revisor debe escribir su dictamen en reviews/ronda_N/review_draft.json.")
    ok, motivo = vigente(run)
    if not ok:
        raise HarnessError(f"audit.json no está vigente ({motivo}).", "Corre audit.py y review.py prepare de nuevo.")
    R = read_json(run / "review.json") or {"rondas": [], "hallazgos": {}}
    hechas = {r["ronda"] for r in R["rondas"]}
    pend = [n for n in _rondas(run) if n not in hechas]
    if not pend:
        raise HarnessError("no hay ronda preparada pendiente.", f"Corre `python scripts/review.py prepare --run {run}`.")
    n = pend[0]
    art = conjunto_revisado(run)
    prep = read_json(run / "reviews" / f"ronda_{n}" / "artefactos.json")
    if prep != art:
        raise HarnessError(f"los artefactos cambiaron después de preparar la ronda {n}.", "Vuelve a correr audit.py y review.py prepare.")
    E = _validar_draft(D, n)
    previos_abiertos = {k for k, v in R["hallazgos"].items() if v["estado"] == "abierto" and not k.startswith("A-")}
    faltan = previos_abiertos - {h["id"] for h in D.get("hallazgos", [])}
    if faltan:
        E.append(f"el dictamen no se pronuncia sobre hallazgos abiertos previos: {sorted(faltan)}")
    hash_art = sha256_json(art)
    for h in D.get("hallazgos", []):
        prev = R["hallazgos"].get(h["id"])
        if h.get("estado") == "resuelto":
            if prev is None:
                E.append(f"{h['id']}: no se puede crear un hallazgo ya resuelto; si no hay problema, no lo reportes")
            elif prev["severidad"] == "bloqueante" and prev["hash_artefactos_apertura"] == hash_art:
                E.append(f"{h['id']}: el bloqueante se marcó resuelto sin que cambiara ningún artefacto desde que se abrió")
    if E:
        raise HarnessError("dictamen inválido:\n" + "\n".join(f"  - {e}" for e in E),
                           "Corrige review_draft.json (ver esquema en scripts/review.py) y vuelve a enviarlo.")

    for h in D.get("hallazgos", []):
        prev = R["hallazgos"].get(h["id"])
        hist = (prev or {}).get("historial", []) + [{"ronda": n, "estado": h["estado"], "fecha": ahora(), "resolucion": h.get("resolucion")}]
        R["hallazgos"][h["id"]] = {**{k: h.get(k) for k in ("severidad", "estado", "evidencia", "artefacto", "accion_requerida", "resolucion")},
                                   "origen": "revisor", "ronda_apertura": (prev or {}).get("ronda_apertura", n),
                                   "hash_artefactos_apertura": (prev or {}).get("hash_artefactos_apertura", hash_art), "historial": hist}
    # fallas de la auditoría determinista
    A = read_json(run / "audit.json")
    activas = {f"A-{x['id'][2:]}" if x["id"].startswith("A-") else x["id"]: x for x in A["verificaciones"]
               if x["veredicto"] == "falla" and x["severidad"] in ("bloqueante", "importante")}
    for hid, x in activas.items():
        prev = R["hallazgos"].get(hid)
        # una falla importante de auditoría que el revisor evaluó y aceptó explícitamente en este dictamen queda aceptada
        dec = next((h for h in D.get("hallazgos", []) if h["id"] == hid), None)
        estado = dec["estado"] if dec and x["severidad"] != "bloqueante" and dec["estado"] == "aceptado" else "abierto"
        R["hallazgos"][hid] = {"severidad": x["severidad"], "estado": estado, "evidencia": x["evidencia"],
                               "artefacto": x.get("artefacto"), "accion_requerida": x.get("accion"),
                               "resolucion": dec.get("resolucion") if dec else None, "origen": "audit.py",
                               "ronda_apertura": (prev or {}).get("ronda_apertura", n),
                               "hash_artefactos_apertura": (prev or {}).get("hash_artefactos_apertura", hash_art),
                               "historial": (prev or {}).get("historial", []) + [{"ronda": n, "estado": estado, "fecha": ahora()}]}
    for hid, v in R["hallazgos"].items():
        if v["origen"] == "audit.py" and hid not in activas and v["estado"] == "abierto":
            v["estado"] = "resuelto"
            v["resolucion"] = "audit.py ya no lo detecta"
            v["historial"].append({"ronda": n, "estado": "resuelto", "fecha": ahora(), "resolucion": v["resolucion"]})
    R["rondas"].append({"ronda": n, "fecha": ahora(), "revisor": D["revisor"], "modelo": D["modelo"], "resumen": D["resumen"],
                        "verificaciones": D["verificaciones"], "hash_artefactos": hash_art, "artefactos": art,
                        "hallazgos_ronda": [h["id"] for h in D.get("hallazgos", [])]})
    atomic_write_json(run / "review.json", R)
    atomic_write_text(run / "review.md", _md(R) + "\n")
    return R


def _md(R: dict) -> str:
    u = R["rondas"][-1]
    ab = [k for k, v in R["hallazgos"].items() if v["estado"] == "abierto" and v["severidad"] == "bloqueante"]
    L = [f"# Revisión adversarial", "", f"Rondas: {len(R['rondas'])}. Última: {u['ronda']} ({u['fecha']}), revisor {u['revisor']} ({u['modelo']}).",
         f"Bloqueantes abiertos: {ab or 'ninguno'}.", "", f"Resumen del revisor: {u['resumen']}", "",
         "## Verificaciones de la última ronda", ""]
    L += [f"- **{v['resultado']}** — {v['nombre']}: {v.get('detalle', '')}" for v in u["verificaciones"]]
    L += ["", "## Hallazgos", "", "| id | origen | severidad | estado | evidencia | artefacto | acción requerida | resolución |",
          "|---|---|---|---|---|---|---|---|"]
    orden = {s: i for i, s in enumerate(SEVERIDADES)}
    for k, v in sorted(R["hallazgos"].items(), key=lambda kv: (kv[1]["estado"] != "abierto", orden[kv[1]["severidad"]])):
        L.append(f"| {k} | {v['origen']} | {v['severidad']} | {v['estado']} | {v['evidencia']} | {v.get('artefacto') or ''} | "
                 f"{v.get('accion_requerida') or ''} | {v.get('resolucion') or ''} |")
    return "\n".join(L)


def criterios(run):
    from audit import conjunto_revisado, vigente
    run = Path(run)
    P = []
    ok, motivo = vigente(run)
    if not ok:
        P.append((f"audit.json no vigente ({motivo}).", f"Corre `python scripts/audit.py --run {run}`."))
    R = read_json(run / "review.json")
    if R is None or not R["rondas"]:
        P.append(("no hay revisión adversarial registrada.", "Lanza al subagente revisor-adversarial (review.py prepare → dictamen → review.py submit)."))
        return P, ["audit.json"]
    if R["rondas"][-1]["hash_artefactos"] != sha256_json(conjunto_revisado(run)):
        P.append(("la última revisión corresponde a una versión anterior de los artefactos.",
                  "Corre audit.py, review.py prepare y lanza una nueva ronda del revisor."))
    ab = [k for k, v in R["hallazgos"].items() if v["estado"] == "abierto" and v["severidad"] == "bloqueante"]
    if ab:
        P.append((f"hay hallazgos bloqueantes abiertos: {ab}.",
                  "Corrige lo que piden (accion_requerida en review.json), vuelve a correr audit.py y lanza una nueva ronda: solo el revisor los cierra."))
    return P, ["audit.json", "review.json"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("accion", choices=["prepare", "submit"])
    ap.add_argument("--run", required=True)
    ap.add_argument("--file")
    a = ap.parse_args()
    if a.accion == "prepare":
        d = prepare(a.run)
        print(f"OK: ronda preparada en {d}. Lanza al subagente revisor-adversarial con: run={a.run}, brief={d / 'brief.md'}.")
        return 0
    if not a.file:
        raise HarnessError("falta --file.", "Pasa la ruta de reviews/ronda_N/review_draft.json.")
    R = submit(a.run, a.file)
    ab = [k for k, v in R["hallazgos"].items() if v["estado"] == "abierto" and v["severidad"] == "bloqueante"]
    print(f"OK: ronda {R['rondas'][-1]['ronda']} registrada en review.json/review.md. Bloqueantes abiertos: {ab or 'ninguno'}.")
    return 1 if ab else 0


if __name__ == "__main__":
    cli(main)
