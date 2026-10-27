"""Pasos reutilizables para llevar un run de prueba de una fase a otra, como lo haría el agente."""
import json
from pathlib import Path

import yaml

from conftest import correr
from experimentos import instalar

FUGAS = {"forecasting": ["ventas_ajustadas", "indice_demanda"], "classification": ["motivo_baja_registrado"],
         "regression": ["y_copia_ruidosa"]}

INTERPRETACION = """

El perfil (eda/profile.json) muestra los datos sin faltantes relevantes salvo los rezagos iniciales,
que el pipeline imputa dentro de cada fold. La figura eda/figuras/target.png muestra la distribución
del target sin colas extremas. La revisión de leakage (leakage_report.json) se atendió excluyendo las
columnas señaladas; quedan como pregunta abierta para negocio los tiempos exactos de publicación de
las variables externas. Riesgo: drift moderado entre mitades del periodo en algunas columnas.
"""

DECLARACION = ("Ningún diagnóstico resultó accionable (diagnostics/diagnostics.json): el sesgo tiene IC que incluye 0, "
               "no hay heterocedasticidad relevante, la importancia por permutación está repartida y las ablaciones "
               "muestran que cada grupo de features aporta más que el margen práctico.\n")


def leer(run, nombre):
    return json.loads((Path(run) / nombre).read_text(encoding="utf-8"))


def editar_contrato(run, fn):
    p = Path(run) / "contract.yaml"
    c = yaml.safe_load(p.read_text(encoding="utf-8"))
    fn(c)
    p.write_text(yaml.safe_dump(c, allow_unicode=True, sort_keys=False), encoding="utf-8")


def agregar_decision(run, num, titulo, texto):
    p = Path(run) / "decisions.md"
    p.write_text(p.read_text(encoding="utf-8") + f"\n## D-{num:03d} — {titulo}\nFecha: 2026-10-22\nDecisión: {texto}\n"
                 f"Justificación: {texto}\n", encoding="utf-8")


def excluir(run, cols, num=2):
    editar_contrato(run, lambda c: c["datos"].setdefault("columnas_excluidas", []).extend(cols))
    agregar_decision(run, num, "Excluir fugas", f"Se excluyen {cols} por hallazgos de leakage_report.json.")


def escribir_interpretacion(run):
    p = Path(run) / "eda_report.md"
    txt = p.read_text(encoding="utf-8")
    cab = "## Interpretación"
    p.write_text(txt.split(cab, 1)[0] + cab + INTERPRETACION, encoding="utf-8")


def justificar_senales(run, texto="El dueño del dato confirmó que se publica antes del cierre del día; validado en la reunión del lunes."):
    """Simula a un experimentador que 'justifica' todas las señales abiertas (así se escapa una fuga)."""
    R = leer(run, "leakage_report.json")
    ab = [x["id"] for x in R["hallazgos"] if x["estado"] == "abierto" and x["clase"] == "senal"]
    (Path(run) / "leakage_resolutions.yaml").write_text(
        yaml.safe_dump({i: {"estado": "justificada", "justificacion": texto} for i in ab}, allow_unicode=True), encoding="utf-8")
    return ab


def hasta_eda(run, tipo, con_fuga=True):
    correr("contract.py", "validate", run=run, check=True)
    if con_fuga:
        excluir(run, FUGAS[tipo])
        correr("contract.py", "validate", run=run, check=True)
    correr("profile_data.py", run=run, check=True)
    correr("leakage_check.py", run=run, check=True)
    escribir_interpretacion(run)
    correr("gate.py", "advance", "eda", run=run, check=True)


def experimentar(run, tipo, ids=("E001", "E002", "E003", "E004", "E005")):
    instalar(run, tipo, list(ids))
    for e in ids:
        correr("experiment_log.py", "run", e, run=run, check=True)
        correr("experiment_log.py", "conclude", e, "--resultado",
               "Resultado observado con cifras en experiments.jsonl, comparado con la referencia declarada.",
               "--decision", "descartada", run=run, check=True)


def hasta_diagnostico(run):
    correr("compare_models.py", run=run, check=True)
    correr("gate.py", "advance", "experimentos", run=run, check=True)
    correr("diagnostics.py", run=run, check=True)
    D = leer(run, "diagnostics/diagnostics.json")
    if not any(x["resultado"] == "accionable" for x in D["resultados"]):
        (Path(run) / "diagnostics" / "declaracion.md").write_text(DECLARACION, encoding="utf-8")
    correr("gate.py", "advance", "diagnostico", run=run, check=True)


def dictamen(run, ronda, hallazgos=(), resumen="Revisión de prueba: re-ejecuté audit.py y recalculé la métrica principal."):
    d = {"ronda": ronda, "revisor": "revisor-adversarial", "modelo": "simulado-en-pruebas", "resumen": resumen,
         "verificaciones": [{"nombre": "re-ejecución de métricas", "resultado": "ok", "detalle": "audit.py --alcance todos"}],
         "hallazgos": list(hallazgos)}
    p = Path(run) / "reviews" / f"ronda_{ronda}" / "review_draft.json"
    p.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
    return p


def ronda_revision(run, hallazgos=(), check_submit=None):
    correr("audit.py", run=run)
    correr("review.py", "prepare", run=run, check=True)
    n = max(int(p.name.split("_")[1]) for p in (Path(run) / "reviews").glob("ronda_*"))
    p = dictamen(run, n, hallazgos)
    return n, correr("review.py", "submit", "--file", p, run=run, check=check_submit)


def hasta_revision(run, tipo, con_fuga=True, ids=("E001", "E002", "E003", "E004", "E005")):
    hasta_eda(run, tipo, con_fuga)
    correr("baselines.py", run=run, check=True)
    correr("gate.py", "advance", "baselines", run=run, check=True)
    experimentar(run, tipo, ids)
    hasta_diagnostico(run)
    ronda_revision(run, check_submit=True)
    correr("gate.py", "advance", "revision", run=run, check=True)
