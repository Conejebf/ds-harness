# ds-harness

Idioma de trabajo, reportes y bitácoras: español (México).

Para cualquier análisis o modelado usa el skill `ds-harness` (`.claude/skills/ds-harness/SKILL.md`):
define las fases, los comandos y los gates. Los subagentes están en `.claude/agents/`
(`experimentador`, `revisor-adversarial`).

Reglas no negociables:
1. Las métricas, estados, comparaciones y dictámenes los escriben los scripts de `scripts/`; nunca a mano.
2. Decisiones de negocio sin definir (`POR_DEFINIR` en contract.yaml) se preguntan al usuario.
3. No abras `_sealed/`. El holdout se evalúa una vez con `scripts/holdout.py`.
4. Si un gate o un hook te bloquea, transmite qué falta y cómo resolverlo; no busques rodeos.
5. Toda decisión no trivial va en `runs/<p>/decisions.md`.

Corre todo con el Python del sistema desde la raíz del repo. Pruebas: `python -m pytest`.
