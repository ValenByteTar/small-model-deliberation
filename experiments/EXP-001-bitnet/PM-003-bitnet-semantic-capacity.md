---
id: PM-003
category: postmortem
status: resolved
created: 2026-08-16
updated: 2026-08-16
author: human
components: [llm_support, bitnet_provider, semantic_ensemble, semantic_adapter, kernel]
tags: [bitnet, llm-support, semantic-evaluation, ensemble, capacity-limit, deprecation, experiment-failure]
related: [ADR-0031, RES-004, RES-016, EXP-010, PM-002]
supersedes: null
superseded_by: null
---

# PM-003 - BitNet-b1.58-2B-4T como Semantic Capability Provider: capacidad insuficiente

## Incident

ADR-0031 (aceptado 2026-08-15) autorizo LLMSupport Fase 1 como observador
paralelo pasivo usando BitNet-b1.58-2B-4T en CPU. La hipotesis original
(RES-004) era que un modelo 2B ternario podria producir hipotesis utiles
observando el runtime sin bloquear el pipeline.

Se ejecutaron tres experimentos progresivos:

1. **Pilot 1 (hipothesis generation)**: BitNet como generador de hipotesis
   GOOD_EVIDENCE / RETRY_RETRIEVAL. Resultado: 0% de utilidad. El modelo
   no seguia el formato few-shot. Produjo 45 RETRY_RETRIEVAL sobre 50 casos
   (90%) sin razonamiento coherente.

2. **Pilot 2 (semantic assessment)**: BitNet como Semantic Capability
   Provider. Tarea replanteada: clasificar relacion claim-evidence como
   SUPPORTS / CONTRADICTS / UNRELATED / PARTIAL. Resultado: 33.3% accuracy
   (4/12), 100% adherencia al protocolo. El modelo sigue el formato pero
   no tiene capacidad semantica suficiente.

3. **Pilot 3 (ensemble de 4 workers)**: 4 instancias de BitNet con prompts
   deliberadamente diferentes (entailment, skeptical, contradiction,
   neutral). Resultado: best single 50%, best ensemble 41.7%. El ensemble
   no supera al mejor worker individual. Alta correlacion de errores
   (Jaccard 0.40-0.64).

**Impacto**: 3 sesiones de experimentacion, ~6 horas de compute, 5.5GB RAM
para el ensemble. Ningun beneficio al pipeline. ADR-0031 queda deprecado.

## Root Cause

La causa raiz **no es** un bug de integracion, ni un problema de prompt,
ni un problema de compilacion de BitNet. La causa raiz es:

**BitNet-b1.58-2B-4T no tiene capacidad semantica suficiente para
evaluar relaciones claim-evidence.**

Evidencia que confirma esta conclusion:

1. **Adherencia al protocolo 100%**: el modelo SI sigue instrucciones de
   formato. Si el problema fuera el prompt o la integracion, el protocolo
   fallaria. No falla.

2. **Accuracy 33% (azar con 4 clases = 25%)**: el modelo apenas supera
   azar. No hay comprension semantica de la relacion claim-evidence.

3. **Reasoning incoherente**: el modelo produce texto que no refleja
   comprension del par claim-evidence. Ej: "CLAIM: This framework applies
   an algorithmic approach EVIDENCE: A system uses al..." — el modelo
   alucina contenido no presente en el input.

4. **Errores sistematicos por tipo**: los casos SUPPORTS son fallados por
   4/4 workers (s-002, s-003). Los casos UNRELATED son los mas faciles
   (u-003: 0/4 wrong). El modelo no reconoce soporte directo pero si
   detecta irrelevancia topica — capacidad parcial, no suficiente.

5. **Ensemble no ayuda**: la diversidad de prompts no produce diversidad
   de errores porque el modelo falla por la misma razon estructural
   (falta de comprension semantica) en los mismos casos. Un ensemble
   no puede compensar una limitacion de capacidad fundamental del modelo.

6. **El modelo es 2B ternario**: BitNet-b1.58-2B-4T usa pesos ternarios
   (-1, 0, 1) en una arquitectura 2B. Esta cuantizacion extrema reduce
   la capacidad de razonamiento semantico. El modelo puede seguir
   patrones de formato pero no puede razonar sobre relaciones semanticas
   entre textos.

## Resolution

1. **ADR-0031 deprecado**: el ADR queda con estado "Deprecated" y
   referencia a este postmortem. La razon: el modelo seleccionado
   (BitNet-b1.58-2B-4T) no cumple el criterio de capacidad minima
   necesario para Fase 1.

2. **LLMSupport desacoplado del pipeline**: el wiring en `bootstrap.py`
   queda comentado con referencia a PM-003 y EXP-010. El codigo de
   LLMSupport, SemanticEnsemble, SemanticAssessmentAdapter se preserva
   como experimento documentado, no como componente activo.

3. **Scripts de pilot preservados**: `run_semantic_pilot.py`,
   `run_ensemble_pilot.py`, `run_llm_support_pilot.py` quedan con
   header explicando que son experimentos deprecados con referencia
   a PM-003 y EXP-010.

4. **No se eliminan los contratos**: `SemanticAssessment` y
   `SEMANTIC_RELATIONS` en `kernel/state.py` se preservan. Si un futuro
   modelo con mayor capacidad (7B+ o MoE) se evalua, los contratos
   ya existen y la frontera arquitectonica esta validada.

## Prevention

1. **No reintentar BitNet-b1.58-2B-4T para tareas semanticas**: este
   postmortem documenta con datos que el modelo no tiene capacidad
   suficiente. Reintentar sin un modelo diferente es desperdicio.

2. **Criterio de capacidad minima para Fase 2**: antes de reactivar
   LLMSupport, cualquier modelo nuevo debe demostrar >60% accuracy en
   el dataset de 12 pares claim-evidence (EXP-010). El dataset esta
   congelado en `scripts/run_semantic_pilot.py`.

3. **Modelo minimo recomendado**: RES-007 sugiere 7B como minimo para
   razonamiento semantico. BitNet-b1.58-2B-4T esta por debajo del
   umbral. Cualquier reevaluacion debe usar un modelo >=7B o un MoE
   con expertos especializados.

4. **Separar "sigue formato" de "comprende semantica"**: el pilot 2
   mostro 100% protocolo pero 33% accuracy. Estos son ortogonales.
   Un modelo puede seguir formato sin comprender semantica. Medir
   ambos por separado.

5. **No asumir que ensemble compensa capacidad**: el pilot 3 confirmo
   que un ensemble de modelos con la misma limitacion de capacidad
   no mejora. La diversidad de prompts no genera diversidad de errores
   cuando el problema es de capacidad, no de perspectiva.
