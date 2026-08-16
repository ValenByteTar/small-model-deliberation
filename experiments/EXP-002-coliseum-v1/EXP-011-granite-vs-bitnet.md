---
id: EXP-011
title: "Granite 4.1 3B Q4 vs BitNet: re-evaluacion de capacidad semantica"
date: 2026-08-16
status: completed
category: experiment
components: [llm_support, semantic_ensemble, ollama_provider, semantic_adapter]
tags: [granite, bitnet, semantic-evaluation, claim-evidence, capacity-eval, cpu]
related: [PM-003, ADR-0031, EXP-010, RES-007]
supersedes: null
superseded_by: EXP-012
---

# EXP-011 - Granite 4.1 3B Q4 vs BitNet

## Hipotesis

Un modelo de 3B parametros con cuantizacion Q4 (Granite 4.1 3B Q4_K_M)
tiene capacidad semantica significativamente mayor que BitNet-b1.58-2B-4T
(2B ternario) para clasificacion claim-evidence, y puede superar el
criterio de >60% accuracy establecido en PM-003.

Hipotesis secundaria: un ensemble de 4 workers Granite (mismo modelo,
prompts diferentes) mejora sobre el mejor worker individual, a diferencia
de BitNet donde el ensemble empeoraba.

## Motivacion

EXP-010 refuto BitNet para SemanticAssessment (33-50% accuracy, 41.7%
ensemble). PM-003 deprecó LLMSupport. Pero los contratos
SemanticAssessment y SemanticAssessmentAdapter están validados
arquitectónicamente. La pregunta es si un modelo más capaz puede
justificar re-habilitar LLMSupport.

RES-007 sugiere evaluar modelos 3B-8B. Granite 4.1 3B Q4 (2.1GB) es
accesible en CPU y significativamente más capaz que BitNet (2B ternario).

## Configuracion

- **Modelo**: ibm/granite4.1:3b-q4_K_M (3.4B params, Q4_K_M, 2.1GB)
- **Runtime**: Ollama, CPU-only (num_gpu=0)
- **Dataset v1**: 12 pares claim-evidence (3 SUPPORTS, 4 CONTRADICTS, 3 UNRELATED, 2 PARTIAL)
- **Single worker**: prompt directo de LLMSupport.semantic_assess()
- **Ensemble**: 4 workers (entailment, skeptical, contradiction, neutral), prompts few-shot del SemanticWorker
- **Aggregator**: ConfidenceWeightedMajorityVote
- **Generation**: num_predict=10, temperature=0.0
- **Hardware**: CPU, 4 threads

## Benchmark

Dataset v1: 12 pares claim-evidence con ground truth conocido.
Mismo dataset que EXP-010 para comparacion directa.

## Resultados

### Single worker (prompt LLMSupport)

| Modelo | Accuracy | Protocol | Latencia |
|--------|----------|----------|----------|
| BitNet-2B (EXP-010 best) | 50.0% (6/12) | 100% | 3.6s |
| Granite 3B Q4 | **91.7% (11/12)** | 100% | 3.5s |

### Ensemble 4 workers

| Modelo | Accuracy | Protocol | Latencia | RAM |
|--------|----------|----------|----------|-----|
| BitNet-2B (EXP-010) | 41.7% (5/12) | 100% | 3.0s | 5539 MB |
| Granite 3B Q4 | **100% (12/12)** | 100% | 9.0s | 4100 MB |

### Comparacion BitNet vs Granite

| Dimension | BitNet-2B | Granite 3B Q4 |
|-----------|-----------|---------------|
| Accuracy single | 33-50% | 91.7% |
| Accuracy ensemble | 41.7% | 100% |
| RAM ensemble | 5539 MB | 4100 MB |
| Latencia single | 3.1-3.7s | 3.5s |
| Latencia ensemble | 3.0s | 9.0s |
| Protocol adherence | 100% | 100% |
| Reasoning quality | Incoherente | Coherente |

## Observaciones

1. Granite supera el criterio PM-003 (>60%) en single (91.7%) y ensemble (100%).
2. El ensemble de Granite mejora sobre single (91.7% -> 100%), a diferencia de BitNet donde empeoraba (50% -> 41.7%).
3. Granite usa menos RAM en ensemble (4.1GB vs 5.5GB) porque comparte una instancia de Ollama.
4. La latencia del ensemble es mayor (9s vs 3s) porque los 4 workers se serializan en una instancia.
5. El unico error de Granite single fue en un caso PARTIAL.

## Anomalias

- El ensemble logro 100% en el dataset v1. Esto puede ser overfitting al
  dataset pequeño (12 casos). Se necesita un benchmark mas exigente para
  confirmar.
- La latencia del ensemble (9s) es alta para uso en produccion.

## Interpretacion

Granite 4.1 3B Q4 tiene capacidad semantica suficiente para
SemanticAssessment. El gap con BitNet es mas que significativo: +41.7%
en single, +58.3% en ensemble. La diferencia no es marginal — es
cualitativa. Granite produce reasoning coherente, BitNet no.

El ensemble funciona con Granite porque los workers tienen errores
decorrelacionados (la diversidad de prompts produce diversidad de
errores). Con BitNet, los errores eran correlacionados (Jaccard 0.40-0.64).

## Decision

- **Hipotesis confirmada**: Granite 3B Q4 supera BitNet significativamente.
- **Hipotesis secundaria confirmada**: el ensemble mejora con Granite.
- **Accion**: Crear benchmark v2 mas exigente (55 casos) para validar.

## Hipotesis refutada

Ninguna — la hipotesis original se confirmo.

## Hipotesis nacida

- H: Un benchmark mas exigente (55 casos, 10 categorias diagnosticas)
  revelara debilidades que el dataset v1 (12 casos) no detecto.
- H: Modelos de 4B params pueden superar a Granite 3B en el benchmark v2.
