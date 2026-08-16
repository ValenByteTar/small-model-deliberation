---
id: EXP-010
title: "BitNet ensemble de 4 evaluadores semanticos: capacidad insuficiente"
date: 2026-08-16
status: completed
category: experiment
components: [llm_support, semantic_ensemble, bitnet_provider, semantic_adapter]
tags: [bitnet, ensemble, semantic-evaluation, claim-evidence, deprecation, capacity-limit]
related: [PM-003, ADR-0031, RES-004, RES-016, EXP-005]
supersedes: null
superseded_by: null
---

# EXP-010 - BitNet ensemble de 4 evaluadores semanticos

## Hypothesis

Un ensemble de 4 instancias de BitNet-b1.58-2B-4T, cada una con un prompt
deliberadamente diferente (entailment, skeptical, contradiction, neutral),
produce una senal semantica (claim-evidence relation) significativamente
mejor que un unico BitNet-b1.58-2B-4T.

Hipotesis secundaria: la diversidad de prompts reduce la correlacion de
errores entre workers, permitiendo que el ensemble compensa errores
individuales.

## Motivation

El pilot anterior (semantic_pilot) mostro que un solo BitNet-b1.58-2B-4T
alcanza 33.3% accuracy en clasificacion claim-evidence (4 clases:
SUPPORTS, CONTRADICTS, UNRELATED, PARTIAL). Eso es apenas sobre azar (25%).
La pregunta era: el problema es el modelo o la tarea? Un ensemble de
evaluadores especializados podria aportar diversidad de perspectiva.

## Configuration

- **Modelo**: BitNet-b1.58-2B-4T (ggml-model-i2_s.gguf)
- **Instancias**: 4, puertos 8081-8084, 1 thread c/u
- **Workers**:
  - A (entailment): logical entailment focus
  - B (skeptical): conservative, reduce false SUPPORTS
  - C (contradiction): active contradiction search
  - D (neutral): balanced independent evaluation
- **Aggregator**: ConfidenceWeightedMajorityVote (deterministic, no LLM)
- **Dataset**: 12 pares claim-evidence con ground truth conocido
  - 3 SUPPORTS, 4 CONTRADICTS, 3 UNRELATED, 2 PARTIAL
- **Generation**: num_predict=10, temperature=0.0
- **Hardware**: CPU, 4 instancias en paralelo

## Metrics

- **accuracy**: relation producida == relation esperada
- **protocol_validity**: relation en {SUPPORTS, CONTRADICTS, UNRELATED, PARTIAL}
- **agreement**: fraccion de workers que votaron la relation final
- **error_jaccard**: |errores_w1 ∩ errores_w2| / |errores_w1 ∪ errores_w2|
- **latency**: tiempo por caso
- **RAM**: memoria total de las 4 instancias

## Results

### Single worker (baseline individual, 4 en paralelo)

| Worker | Role | Accuracy | Protocol | Avg lat | Distribution |
|--------|------|----------|----------|---------|--------------|
| A | entailment | 33.3% (4/12) | 100% | 3.1s | PARTIAL:7, CONTRADICTS:2, UNRELATED:3 |
| B | skeptical | 16.7% (2/12) | 100% | 3.2s | UNRELATED:6, CONTRADICTS:4, SUPPORTS:2 |
| C | contradiction | 33.3% (4/12) | 100% | 3.7s | UNRELATED:8, CONTRADICTS:3, PARTIAL:1 |
| D | neutral | 50.0% (6/12) | 100% | 3.6s | CONTRADICTS:7, UNRELATED:4, PARTIAL:1 |

### Ensemble

| Ensemble | Accuracy | Protocol | Avg lat | Avg agreement |
|----------|----------|----------|---------|---------------|
| 2_AB | 33.3% (4/12) | 100% | 2.2s | 0.62 |
| 2_CD | 41.7% (5/12) | 100% | 2.5s | 0.75 |
| 3_ABC | 25.0% (3/12) | 100% | 2.7s | 0.64 |
| 4_ABCD | 41.7% (5/12) | 100% | 3.0s | 0.62 |

### Correlacion de errores (Jaccard)

| Par | Both wrong | Jaccard |
|-----|-----------|---------|
| A-B | 7 | 0.64 |
| B-C | 7 | 0.64 |
| A-C | 6 | 0.60 |
| C-D | 5 | 0.56 |
| B-D | 5 | 0.46 |
| A-D | 4 | 0.40 |

### Por caso (workers wrong)

| Caso | Expected | Wrong |
|------|----------|-------|
| s-002 | SUPPORTS | 4/4 |
| s-003 | SUPPORTS | 4/4 |
| p-002 | PARTIAL | 4/4 |
| s-001 | SUPPORTS | 3/4 |
| c-003 | CONTRADICTS | 3/4 |
| p-001 | PARTIAL | 3/4 |
| p-003 | CONTRADICTS | 3/4 |
| c-001 | CONTRADICTS | 2/4 |
| c-002 | CONTRADICTS | 2/4 |
| u-001 | UNRELATED | 2/4 |
| u-002 | UNRELATED | 2/4 |
| u-003 | UNRELATED | 0/4 |

### Resources

- RAM: 5539 MB (4 instancias, ~1.6GB c/u)
- Latency: avg 3.0s/case ensemble de 4

## Conclusion

**Hipotesis refutada.** El ensemble de 4 workers no produce una senal
semantica significativamente mejor que un unico BitNet-b1.58-2B-4T.

Datos clave:

1. **Best single (D: 50%) > best ensemble (4_ABCD: 41.7%)**. El ensemble
   promedia hacia abajo. Arrastra al mejor worker.

2. **Alta correlacion de errores** (Jaccard 0.40-0.64). Los workers fallan
   en los mismos casos. 3 de 12 casos tienen 4/4 workers wrong. La
   diversidad de prompts no produjo diversidad de errores.

3. **Errores sistematicos por tipo**: SUPPORTS fallado por 4/4 workers
   (s-002, s-003). El modelo no reconoce soporte directo. UNRELATED es
   lo mas facil (u-003: 0/4 wrong). El problema es de capacidad semantica
   del modelo, no de diversidad de perspectiva.

4. **Ensemble de 3 (ABC: 25%) peor que cualquier single**. La votacion
   ponderada sin el mejor worker degrada.

**Causa raiz**: BitNet-b1.58-2B-4T (2B ternario) no tiene capacidad
semantica suficiente para evaluar relaciones claim-evidence. Un ensemble
de modelos con la misma limitacion de capacidad no puede compensar esa
limitacion. Ver PM-003 para analisis completo.

## Recommendation

- [x] Nothing — experimento fallido, no promocionar
- [ ] Benchmark — no congela como referencia (no hay valor)
- [ ] ADR — ADR-0031 se deprecada por PM-003, no se promueve
- [ ] Decision — no DEC

**Accion tomada**: ADR-0031 deprecado (PM-003). LLMSupport desacoplado
del pipeline. Codigo preservado como experimento documentado. No
reintentar BitNet-b1.58-2B-4T para tareas semanticas.

**Reevaluacion futura**: si se evalua un modelo >=7B (RES-007), usar el
mismo dataset de 12 pares. Criterio de aprobacion: >60% accuracy.
Los contratos SemanticAssessment y SemanticAssessmentAdapter ya existen
y la frontera arquitectonica esta validada.
