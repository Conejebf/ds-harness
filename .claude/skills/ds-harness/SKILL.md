---
name: ds-harness
description: Orquesta un análisis o modelado de ciencia de datos (forecasting, clasificación o regresión tabular) con fases, artefactos verificables y gates que fallan por código. Úsala para cualquier estudio de modelado en este repo, para retomar uno ("¿en qué fase va runs/x?") o cuando pidan entrenar, comparar o elegir un modelo.
---

# ds-harness — orquestador

Los controles de calidad son CÓDIGO: cada fase produce artefactos que `scripts/gate.py` verifica
por hash. Si una fase no produjo su artefacto, la siguiente no arranca y el reporte no se emite.
Tú orquestas; los scripts calculan. Nunca escribas métricas, estados ni dictámenes a mano.

Todos los comandos se corren desde la raíz del repo con el Python del sistema.
`<run>` = `runs/<proyecto>`. En cualquier momento: `python scripts/gate.py status --run <run>`.

## Fase 0 — Contrato
1. `python scripts/contract.py init --run <run> --type <forecasting|classification|regression>`
2. Copia el snapshot a `<run>/data/` (CSV o parquet; si viene de SQL, expórtalo y anota la query en `datos.procedencia`).
   Puedes perfilarlo antes de decidir: `python scripts/profile_data.py --run <run> --exploratorio`.
3. Completa `contract.yaml`. Todo `POR_DEFINIR` que sea decisión de negocio (target, momento de
   predicción, horizonte, generalización, métrica, margen práctico, restricciones) **se pregunta al
   usuario**; no lo inventes. Declara la disponibilidad de cada columna.
4. `python scripts/contract.py validate --run <run>` → congela snapshot, separa y sella el holdout, materializa splits.

## Fase 1 — EDA y leakage
1. `python scripts/profile_data.py --run <run>`
2. `python scripts/leakage_check.py --run <run>` → si `bloqueado`: excluye features (contract.yaml +
   decisions.md + re-validar) o justifica SEÑALES en `leakage_resolutions.yaml`. Las violaciones no se justifican.
3. Escribe la sección `## Interpretación` de `eda_report.md` citando artefactos.
4. `python scripts/gate.py advance eda --run <run>`

## Fase 2 — Escalera de baselines
`python scripts/baselines.py --run <run>` y `python scripts/gate.py advance baselines --run <run>`.

## Fase 3 — Experimentación
Lanza el subagente **experimentador** con la ruta del run (o sigue su procedimiento tú mismo).
Cierre: `python scripts/compare_models.py --run <run>` y `python scripts/gate.py advance experimentos --run <run>`.

## Fase 4 — Diagnóstico
`python scripts/diagnostics.py --run <run>`. Si nada es accionable, escribe `diagnostics/declaracion.md`
explicando por qué. Si algo es accionable y cambia el modelo, vuelve a fase 3 con un experimento nuevo.
`python scripts/gate.py advance diagnostico --run <run>`.

## Fase 5 — Revisión adversarial
1. `python scripts/audit.py --run <run>`
2. `python scripts/review.py prepare --run <run>` → crea `reviews/ronda_N/brief.md`.
3. Lanza el subagente **revisor-adversarial** en contexto NUEVO pasándole SOLO: la ruta del run y
   la del brief. No le pases tu razonamiento, resúmenes ni conclusiones.
4. Él envía su dictamen con `review.py submit`. Si hay bloqueantes: corrige (nuevo experimento,
   exclusión, etc.), re-audita, prepara otra ronda y vuelve a lanzar al revisor. Tú no cierras bloqueantes.
5. `python scripts/gate.py advance revision --run <run>`

## Fase 6 — Holdout (una sola vez)
`python scripts/holdout.py evaluate --run <run>`. Un resultado decepcionante NO autoriza probar otro
modelo contra el mismo holdout; una nueva iteración exige `contract.py validate --new-version`.

## Fase 7 — Reporte
`python scripts/final_report.py --run <run>`. Se niega si falta algo o hay bloqueantes abiertos.

## Reglas de comportamiento
- Nunca declares un modelo "bueno" sin compararlo contra el ingenuo y la escalera (comparison.md).
- Nunca reportes una métrica sin su intervalo o su variabilidad entre folds.
- Si algo es sospechosamente bueno, asume leakage hasta demostrar lo contrario.
- Si los datos no alcanzan para una conclusión, dilo (estado `inconcluso`/`operativa`); no la fuerces.
- Cuando un gate falla, transmite al usuario QUÉ falta y CÓMO resolverlo (el script lo imprime).
- Toda decisión no trivial va en `decisions.md` (`## D-NNN — título`, Decisión, Justificación).
- No abras `_sealed/`; no edites artefactos generados (los hooks lo bloquean). Si un hook te
  bloquea, no busques rodeos: usa el script correspondiente.
