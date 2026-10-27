# ds-harness

Harness de ciencia de datos para Claude Code: fases con artefactos verificables y **gates que
fallan por código**. Si una fase no produjo su artefacto, la siguiente no arranca y el reporte final
no se emite. Cubre forecasting, clasificación y regresión tabular (pandas + scikit-learn + statsmodels).

## Requisitos
Python 3.11 con `pandas`, `numpy`, `scipy`, `scikit-learn`, `statsmodels`, `pyyaml`, `matplotlib`
(ver `requirements.txt`). Opcionales: `pyarrow` (si no está, las tablas internas se guardan en
pickle), `xgboost`, `lightgbm`. Los tests ordinarios no requieren red, LLM ni librerías opcionales.

## Uso en un proyecto nuevo
Abre Claude Code con este repo como raíz (así cargan `.claude/settings.json`, los hooks, el skill
`ds-harness` y los subagentes) y pide el estudio; el skill guía las fases. A mano:

```
python scripts/contract.py init --run runs/mi_estudio --type forecasting
#   copiar el snapshot a runs/mi_estudio/data/ y completar contract.yaml (POR_DEFINIR = preguntar)
python scripts/profile_data.py --run runs/mi_estudio --exploratorio      # opcional, antes del contrato
python scripts/contract.py validate --run runs/mi_estudio                 # fase 0
python scripts/profile_data.py --run runs/mi_estudio                      # fase 1
python scripts/leakage_check.py --run runs/mi_estudio
#   escribir "## Interpretación" en eda_report.md
python scripts/gate.py advance eda --run runs/mi_estudio
python scripts/baselines.py --run runs/mi_estudio                         # fase 2
python scripts/gate.py advance baselines --run runs/mi_estudio
#   fase 3: experiments/E0xx.py (copiar _plantilla.py) por hipótesis
python scripts/experiment_log.py run E001 --run runs/mi_estudio
python scripts/experiment_log.py conclude E001 --resultado "..." --decision descartada --run runs/mi_estudio
python scripts/compare_models.py --run runs/mi_estudio
python scripts/gate.py advance experimentos --run runs/mi_estudio
python scripts/diagnostics.py --run runs/mi_estudio                       # fase 4
python scripts/gate.py advance diagnostico --run runs/mi_estudio
python scripts/audit.py --run runs/mi_estudio                             # fase 5
python scripts/review.py prepare --run runs/mi_estudio
#   subagente revisor-adversarial (contexto limpio) → review_draft.json → review.py submit
python scripts/gate.py advance revision --run runs/mi_estudio
python scripts/holdout.py evaluate --run runs/mi_estudio                  # fase 6, una sola vez
python scripts/final_report.py --run runs/mi_estudio                      # fase 7
python scripts/gate.py status --run runs/mi_estudio                       # en cualquier momento
```

Demo con datos sintéticos (fugas plantadas): `python tests/fixtures/synth.py --tipo forecasting --out runs/demo`.

## Fases, artefactos y gates
| fase | artefactos | la aprueba |
|---|---|---|
| 0 contrato | contract.yaml, snapshot.json, dev, _sealed/holdout, holdout_manifest.json, splits.json | `contract.py validate` |
| 1 eda | eda/profile.json, eda_report.md (con interpretación), leakage_report.json | `gate.py advance eda` |
| 2 baselines | baselines.json (ingenuo, lineal, GBM) | `gate.py advance baselines` |
| 3 experimentos | experiments.jsonl, preds/, experiments/_executed/, comparison.json | `gate.py advance experimentos` |
| 4 diagnóstico | diagnostics/diagnostics.json (+ declaracion.md si nada es accionable) | `gate.py advance diagnostico` |
| 5 revisión | audit.json, review.json/review.md | `gate.py advance revision` |
| 6 holdout | holdout_freeze.json, holdout_result.json | `holdout.py evaluate` |
| 7 reporte | final_report.md | `final_report.py` |

Cada fase guarda en `state.json` el hash de sus artefactos y de las fases anteriores: cambiar un
artefacto o re-aprobar una fase previa invalida las posteriores. Cada gate que falla imprime qué
falta y cómo resolverlo, y sale con código ≠ 0.

## Qué está implementado (y probado)
- Contrato: completitud, `POR_DEFINIR` bloquea, coherencia problema↔validación (tiempo ⇒ temporal
  con gap; entidades nuevas ⇒ grupos; nunca KFold aleatorio con dependencia), disponibilidad por feature.
- Holdout sellado desde la fase 0, evaluado una vez, con congelamiento previo del candidato,
  recuperación registrada y nueva versión obligatoria si se quiere iterar.
- Leakage por evidencia: violaciones comprobadas (bloquean, no se justifican) vs señales (se
  investigan). Pruebas deterministas de identidad y desfase exacto con el target.
- Runner: el harness calcula todas las métricas, ajusta todo dentro de cada fold, valida alineación
  y cobertura, guarda predicciones fuera de muestra, registra fallos y nunca sobrescribe experimentos.
- Comparación: Nadeau–Bengio (kfold), bootstrap pareado por entidad (grupos) o por bloques temporales
  (forecasting), Holm sobre una familia explícita, margen práctico, elección determinista del más
  simple, estados `concluyente | provisional | operativa | override`. Cada candidato reporta el IC de
  su propia métrica con la unidad de remuestreo del diseño (la desviación entre folds puede subestimar
  la incertidumbre, p. ej. con folds estratificados por grupos).
- Diagnóstico estructurado (`accionable | sin_hallazgos | no_evaluable`), auditoría determinista
  (reconstrucción de splits, reproducción y re-ejecución de métricas con tolerancias, prueba del lote,
  linaje "viaje en el tiempo"), revisión por rondas con bloqueo ante bloqueantes abiertos.
- Hooks `PreToolUse` que bloquean la edición directa de artefactos generados y el acceso a `_sealed/`.

## Límites (léelos)
- **Integridad ≠ seguridad.** Los hashes detectan cambios accidentales o inconsistentes; quien tenga
  permisos completos puede reescribir artefactos y hashes. Los hooks son heurísticos (un comando
  ofuscado los evade).
- **Holdout = aislamiento operativo.** Los scripts no lo cargan, pero el snapshot original y
  `_sealed/` siguen en disco. No es inaccesible.
- **El revisor tiene Bash**: no es técnicamente de solo lectura, y un dictamen podría falsificarse
  desde la misma sesión. Su valor depende de lanzarlo con contexto limpio.
- **Leakage sin linaje.** Una feature del futuro disfrazada (con ruido o transformada) y sin linaje
  declarado no siempre es detectable: se reporta `no_verificable` + señal estadística, y queda en
  manos del revisor.
- **Inferencia.** Holm controla la familia "vs el mejor"; no corrige el sesgo de búsqueda adaptativa.
  Con pocas unidades de remuestreo el resultado es `inconcluso` / elección `operativa`.
- **Reproducibilidad** con tolerancias explícitas (`auditoria.tolerancia_*`), no bit a bit entre plataformas.
- Clasificación multiclase: las métricas están verificadas contra scikit-learn, pero el recorrido
  completo (runner → diagnóstico → holdout) solo se probó con clasificación binaria.
- Tiempo: se asume grilla regular (el periodo es el rango de los valores únicos de la columna de tiempo).

## Extensiones pendientes
Aislamiento del holdout por permisos/proceso separado; evaluación anidada para afirmaciones
confirmatorias tras mucha selección; firma de dictámenes del revisor; soporte nativo de polars y
de LightGBM; ejecución en paralelo.

## Pruebas
```
python -m pytest                       # deterministas (sin red ni LLM); ~30 min
python -m pytest -m llm                # revisor real vía `claude -p` (cuesta tokens)
```
