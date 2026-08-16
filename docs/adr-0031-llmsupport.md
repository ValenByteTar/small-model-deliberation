# ADR-0031 - LLMSupport Fase 1: observador paralelo pasivo

- **Estado:** Deprecated (PM-003, 2026-08-16)
- **Fecha:** 2026-08-15
- **Relaciona con:** RES-004, RES-016, ADR-0005, ADR-0007, ADR-0014, ADR-0020
- **Depende de:** ADR-0005 (Observability), ADR-0007 (ModelProvider), ADR-0014 (Inyeccion de dependencias)

> **DEPRECADO**: BitNet-b1.58-2B-4T no tiene capacidad semantica suficiente
> para producir hipotesis utiles ni evaluar relaciones claim-evidence.
> Tres experimentos progresivos (hipothesis generation, semantic assessment,
> ensemble de 4 workers) confirmaron accuracy 33-50%, alta correlacion de
> errores y razonamiento incoherente. LLMSupport esta desacoplado del
> pipeline. El codigo se preserva como experimento documentado.
>
> Ver:
> - knowledge/postmortems/PM-003-bitnet-semantic-capacity-insufficient.md
> - knowledge/experiments/EXP-010-bitnet-ensemble-semantic-capacity.md
>
> **Reactivacion futura**: si se evalua un modelo >=7B (RES-007) que supere
> >60% accuracy en el dataset de EXP-010, se puede reactivar el wiring en
> bootstrap.py. Los contratos (SemanticAssessment, SemanticAssessmentAdapter)
> ya existen y la frontera arquitectonica esta validada.

## Contexto

RES-004 propone un componente LLMSupport que corre paralelo al pipeline, observa
eventos del runtime y produce hipotesis sin bloquear ni decidir. RES-004 §15
indica que el research prepara la promocion futura a ADR; RES-004 §16 lista los
criterios de promocion.

RES-004 §9 describe Fase 1 (observabilidad pura, pasiva):
- Nada cambia. El pipeline ejecuta exactamente igual.
- Se mide precision y recall de hipotesis.
- Recien despues se habilita influencia (Fase 2, requiere ADR separado).

Esto cumple P17 (ADR-0020): la observabilidad no cambia el comportamiento.

El modelo BitNet-b1.58-2B-4T fue verificado en CPU con bitnet.cpp:
- 121.5 t/s prompt processing, 21.7 t/s generation (4 threads)
- ~1.1GB RAM, 0% GPU
- No compite con el modelo principal del pipeline (Ollama en GPU)

## Decision

Implementar LLMSupport Fase 1 (modo passive) como componente transversal de
observabilidad:

1. **Hypothesis contract**: nuevo tipo de output (suggestion, confidence,
   reasoning, stage, run_id) — distinto de EvaluationSignal y ActionDecision
2. **BitNetModelProvider**: implementacion de ModelProvider (ADR-0007) que
   habla con llama-server.exe corriendo BitNet en CPU
3. **LLMSupport**: componente que se suscribe a TraceEvent, usa
   BitNetModelProvider para generar hipotesis, las loggea (passive mode)
4. **FanOutTraceSink**: wrapper que fan-out eventos a multiples sinks
5. **Feature flag**: `knowledge.llm_support_enabled` (default: false)
6. **Modo**: passive (solo log), advisory (Fase 2, requiere ADR separado), off

### Limites de Fase 1 (passive)

- LLMSupport **nunca bloquea** el pipeline (corre en thread separado)
- LLMSupport **nunca decide** (no invoca capabilities, no modifica
  ExecutionState, no sobrescribe decisiones)
- LLMSupport **nunca reemplaza** ASSESS, VERIFY ni al Policy Engine
- LLMSupport **solo observa** y produce hipotesis que se loggean
- Las hipotesis **no llegan** al Policy Engine en Fase 1

### Frontera con Fase 2 (advisory)

El modo advisory (Fase 2) cruza P17: LLMSupport deja de ser observabilidad pura
y pasa a influir el control. Habilitarlo exige:
- ADR separado
- Medicion previa de precision/recall de Fase 1
- Gate de precision/recall (P4, P16)

Este ADR **solo autoriza Fase 1 (passive)**. Fase 2 queda explicitamente fuera
de alcance.

## Consecuencias

- **Nueva superficie**: BitNetModelProvider, LLMSupport, FanOutTraceSink,
  Hypothesis contract
- **No cambia comportamiento del pipeline**: Fase 1 es observabilidad pura (P17)
- **Modelo dedicado en CPU**: BitNet no compite con GPU del pipeline principal
- **Coste**: ~1.1GB RAM adicional, 1 thread pool para LLMSupport
- **Medible**: las hipotesis loggeadas permiten calcular precision/recall antes
  de habilitar Fase 2

## Compliance

| Principio | Como cumple |
|-----------|-------------|
| P17 (ADR-0020) | Observabilidad no cambia comportamiento — Fase 1 es passive |
| P16 (ADR-0020) | Ownership de decisiones — LLMSupport produce opiniones, Policy Engine decide |
| P14 | Una responsabilidad por eslabon — LLMSupport observa, no ejecuta |
| P13 (ADR-0014) | Inyeccion de dependencias — ModelProvider inyectado |
| P9 | Determinismo en control, razonamiento en lenguaje — LLMSupport razona, no controla |
| P3 (ADR-0005) | Observabilidad antes que magia — toda hipotesis es trazable |
| P4 | Medible antes que inteligente — pasivo primero, medir, luego habilitar influencia |
| P2 | Contratos estables, implementaciones desechables — ModelProvider inyectado, modelo reemplazable |
| ADR-0011 | Local-first — BitNet corre local en CPU, sin dependencias externas |
