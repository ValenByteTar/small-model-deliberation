---
id: EXP-012
title: "Coliseo v1: 3 modelos vs SemanticAssessment Benchmark v2 (CPU)"
date: 2026-08-16
status: completed
status_note: "RESULTADOS DE LLAMA3.2 CONTAMINADOS — ver POST-001. Granite y Qwen3 no afectados."
category: experiment
components: [semantic_ensemble, ollama_provider, semantic_adapter]
tags: [granite, llama, qwen, semantic-evaluation, benchmark-v2, cpu, ensemble, coliseo, protocol-contamination]
related: [PM-003, ADR-0031, EXP-010, EXP-011, RES-007, POST-001]
supersedes: EXP-011
superseded_by: EXP-013
---

# EXP-012 - Coliseo v1: 3 modelos vs Benchmark v2 (CPU)

> **ADVERTENCIA (POST-001):** Los resultados de Llama3.2 en este
> experimento estan **contaminados por defectos de protocolo**
> (`num_predict=10` + parser leniento que defaultea a `UNRELATED`).
> El 16.4% reportado es un artefacto, no una medida de capacidad
> semantica. La repeticion con protocolo corregido da **58.2% single**
> (ver POST-001). Los resultados de Granite y Qwen3 no estan afectados
> porque sus modelos no truncaban con `num_predict=10`. La conclusion
> de "Llama3.2 es semanticamente insuficiente" ha sido **retirada**.

## Hipotesis

Modelos de 3-4B parametros con cuantizacion Q4 pueden alcanzar >60%
accuracy en un benchmark semantico exigente (55 casos, 10 categorias
diagnosticas), y el ensemble de 4 workers mejora sobre single.

Hipotesis secundaria: modelos de 3B (Llama 3.2 3B) tienen capacidad
semantica insuficiente, similar a BitNet.

## Motivacion

EXP-011 mostro Granite 3B Q4 con 91.7% single y 100% ensemble en el
dataset v1 (12 casos). Pero 12 casos es demasiado pequeño para confiar.
Se necesita un benchmark mas exigente con categorias diagnosticas
especificas (paraphrase, negation, implicit contradiction, over-specificity,
wrong subject, wrong context, adversarial) para detectar debilidades
reales.

Adicionalmente, se quiere comparar Granite contra otros modelos del
mismo rango (Llama 3.2 3B, Qwen3 4B-RAG) para identificar el mejor
modelo para SemanticAssessment.

## Configuracion

- **Modelos**:
  - ibm/granite4.1:3b-q4_K_M (3.4B, Q4_K_M, 2.1GB)
  - llama3.2:3b (3B, Q4_K_M, 2.0GB)
  - qwen3-4b-rag:latest (4B, Q4_K_M, 2.5GB)
- **Runtime**: Ollama, CPU-only (num_gpu=0)
- **Benchmark v2**: 55 casos, 10 categorias diagnosticas
  - direct_evidence (6), paraphrase (6), partial_support (6),
    explicit_contradiction (5), implicit_contradiction (5),
    over_specificity (5), negation (6), wrong_subject (5),
    wrong_context (5), adversarial (6)
- **Distribucion esperada**: SUPPORTS:12, PARTIAL:18, CONTRADICTS:16, UNRELATED:9
- **Configs**: single (neutral), ensemble_2 (entailment+skeptical), ensemble_4 (4 roles)
- **Prompts**: few-shot del SemanticWorker (4 ejemplos cada uno)
- **Aggregator**: ConfidenceWeightedMajorityVote
- **Generation**: num_predict=10, temperature=0.0, num_thread=4
- **Hardware**: CPU
- **Total evaluaciones**: 495 (3 modelos x 3 configs x 55 casos)

## Benchmark

semantic_assessment_benchmark_v2.json — 55 casos con categorias
diagnosticas diseñadas para testing riguroso de capacidad semantica.

## Resultados

### Accuracy global

| Modelo | single | ensemble_2 | ensemble_4 |
|--------|--------|------------|------------|
| granite-3b-q4 | 61.8% (34/55) | 67.3% (37/55) | 76.4% (42/55) |
| llama32-3b | 16.4% (9/55) | 29.1% (16/55) | 20.0% (11/55) |
| **qwen3-4b-rag** | **78.2% (43/55)** | **83.6% (46/55)** | 81.8% (45/55) |

### Latencia

| Modelo | single | ensemble_2 | ensemble_4 |
|--------|--------|------------|------------|
| granite-3b-q4 | 3.5s | 4.7s | 6.4s |
| llama32-3b | 3.8s | 4.7s | 6.1s |
| qwen3-4b-rag | 4.2s | 5.6s | 8.4s |

### Wall time

| Modelo | single | ensemble_2 | ensemble_4 |
|--------|--------|------------|------------|
| granite-3b-q4 | 195s | 257s | 352s |
| llama32-3b | 207s | 260s | 336s |
| qwen3-4b-rag | 230s | 308s | 465s |

### Ensemble vs Single (delta)

| Modelo | single | ens2 delta | ens4 delta |
|--------|--------|-----------|-----------|
| granite-3b-q4 | 61.8% | +5.4% | **+14.5%** |
| llama32-3b | 16.4% | +12.7% | +3.6% |
| qwen3-4b-rag | 78.2% | **+5.5%** | +3.6% |

### Accuracy por categoria (mejor config: qwen3-4b-rag ensemble_2)

| Categoria | Accuracy |
|-----------|----------|
| direct_evidence | 100% |
| paraphrase | 100% |
| explicit_contradiction | 100% |
| wrong_subject | 20% (ensemble_2) / 100% (single) |
| negation | 83% |
| implicit_contradiction | 60% |
| over_specificity | 60% |
| partial_support | 67% |
| wrong_context | 40% |
| adversarial | 67% |

## Observaciones

1. **Qwen3 4B-RAG es el ganador**: 78.2% single, 83.6% ensemble_2.
   Supera el criterio PM-003 (>60%) en todas las configs.
2. **Granite 3B Q4 cae del 91.7% (v1) al 61.8% (v2)**: el benchmark v2
   es significativamente mas exigente. El dataset v1 era demasiado facil.
3. **Llama 3.2 3B es catastrofico**: 16.4% single, peor que azar (25%).
   No acierta ni direct_evidence (0/6). Descartado.
4. **Ensemble_2 es el sweet spot para Qwen3**: +5.5% sobre single,
   sin el overhead de ensemble_4. Ensemble_4 no aporta sobre ensemble_2.
5. **Granite se beneficia mas del ensemble_4**: +14.5% sobre single.
   Su single es debil pero los 4 roles se complementan.
6. **partial_support y wrong_context son dificiles para todos**:
   ningun modelo supera 67% en partial_support ni 60% en wrong_context.
7. **wrong_subject colapsa en ensemble_2 para Qwen3**: 100% single ->
   20% ensemble_2. Los workers especializados ignoran el sujeto y se
   enfocan en similitud de contenido.

## Anomalias

- Granite single v1 = 91.7%, v2 = 61.8%. Delta -29.9%. El dataset v1
  era demasiado optimista. Esto valida la creacion del benchmark v2.
- Qwen3 wrong_subject: 100% single -> 20% ensemble_2. El ensemble
  destruye una categoria donde el single era perfecto. Los workers
  especializados introducen sesgos que un modelo fuerte no tiene.
- Llama 3.2 3B solo acierta wrong_context (80%) — probablemente porque
  clasifica todo como UNRELATED por default.
- Granite ensemble_2 en GPU (EXP-013) dio 78.2% vs 67.3% en CPU.
  Delta +10.9% inexplicable por cambio de hardware solo. Posible
  no-determinismo del scheduler de Ollama.

## Interpretacion

El benchmark v2 revela que la capacidad semantica de modelos 3-4B es
real pero limitada. Qwen3 4B-RAG (83.6% ensemble_2) es el mejor modelo
evaluado, pero tiene debilidades especificas:

- **wrong_subject**: el fine-tuning RAG ayuda en single (100%) pero el
  ensemble lo destruye (20%). El ensemble no es universalmente beneficioso.
- **partial_support**: la linea entre SUPPORTS y PARTIAL es intrinsecamente
  ambigua. Ningun modelo la resuelve consistentemente.
- **wrong_context**: distinguir "mismo concepto, diferente contexto" de
  "evidencia relevante" requiere razonamiento sobre el dominio.

El ensemble ayuda a modelos debiles (Granite +14.5%) pero puede
perjudicar a modelos fuertes en categorias especificas (Qwen3
wrong_subject -80%). El ensemble no es un pipeline universalmente
beneficioso — depende del modelo y la categoria.

## Decision

- **Hipotesis confirmada**: Qwen3 y Granite superan >60% en v2.
- ~~**Hipotesis secundaria confirmada**: Llama 3.2 3B no tiene capacidad
  semantica suficiente (16.4%, peor que azar).~~ **RETIRADA (POST-001)**:
  El 16.4% era un artefacto del protocolo (`num_predict=10` + parser
  leniento). La repeticion con protocolo corregido da 58.2% single.
  Llama3.2 tiene capacidad limitada pero real, no "insuficiente".
- **Ganador**: Qwen3 4B-RAG ensemble_2 = 83.6%
- **Accion**: Re-evaluar en GPU para medir speedup (EXP-013).
  Explorar deliberacion entre workers (EXP-014).
  **Re-evaluar Llama3.2 con protocolo corregido (POST-001).**

## Hipotesis refutada

- H(explicit): el ensemble_4 siempre mejora sobre ensemble_2.
  Refutada para Qwen3: ensemble_4 (81.8%) < ensemble_2 (83.6%).
- H(explicit): el ensemble siempre mejora sobre single.
  Refutada para Qwen3 wrong_subject: 100% single -> 20% ensemble_2.

## Hipotesis nacida

- H: La GPU no cambia la calidad de las inferencias, solo la velocidad.
- H: La deliberacion entre workers (debate + judge) puede corregir
  errores que el ensemble paralelo no puede, especialmente en
  partial_support y wrong_context.
- H: El ensemble destruye wrong_subject porque los workers especializados
  ignoran la entidad. Un worker dedicado a context/entity (Worker D)
  podria prevenir esto si tiene peso suficiente.
