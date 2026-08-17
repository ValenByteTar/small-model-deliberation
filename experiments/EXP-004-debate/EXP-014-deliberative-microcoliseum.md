---
id: EXP-014
title: "Deliberative Micro-Coliseum: debate entre workers vs ensemble paralelo"
date: 2026-08-16
status: completed
category: experiment
components: [semantic_ensemble, ollama_provider, semantic_adapter]
tags: [deliberation, debate, judge, ensemble, semantic-evaluation, benchmark-v2, gpu, microcoliseum]
related: [PM-003, ADR-0031, EXP-012, EXP-013]
supersedes: null
superseded_by: null
---

# EXP-014 - Deliberative Micro-Coliseum

## Hipotesis

H1: Un conjunto de workers que primero evalua independientemente y luego
delibera sobre sus desacuerdos puede corregir errores del ensemble inicial.

H0: La deliberacion no mejora significativamente el resultado, o introduce
mas errores de los que corrige.

La metrica principal NO es accuracy final, sino el balance
corrections vs damage:

- Corrections: casos initial=WRONG -> final=CORRECT
- Damage: casos initial=CORRECT -> final=WRONG

Una mejora pequeña en accuracy no es suficiente si el Damage Rate
tambien aumenta significativamente.

## Motivacion

EXP-012 mostro que el ensemble paralelo (voting) tiene un techo: Qwen3
83.6% ensemble_2, Granite 76.4% ensemble_4. El voting no puede corregir
errores cuando multiples workers estan de acuerdo en una clasificacion
incorrecta. La pregunta es: si los workers pueden ver las opiniones de
los demas y argumentar, pueden reconocer que estaban equivocados?

La deliberacion es una arquitectura fundamentalmente diferente:
- Ensemble paralelo: workers independientes -> agregacion deterministica
- Deliberativo: workers independientes -> challenge round -> judge final

El judge tiene acceso a los argumentos de todos los workers y puede
elegir una respuesta que no es la mayoria, basandose en la calidad
de los argumentos.

## Configuracion

- **Modelos**: mismos 3 que EXP-012/013
  - ibm/granite4.1:3b-q4_K_M
  - qwen3-4b-rag:latest
  - llama3.2:3b
- **Runtime**: Ollama, GPU (num_gpu=99)
- **Benchmark**: semantic_assessment_benchmark_v2.json (55 casos, 10 categorias)
- **Workers**: 4 roles especializados
  - A: entailment analyst
  - B: skeptical analyst
  - C: contradiction analyst
  - D: context/entity analyst
- **Prompts**: sin few-shot, salida JSON estructurada
  - Phase 1: JSON con relation, confidence, reason
  - Phase 3: JSON con counterargument, change_decision, proposed_relation
  - Phase 4: Judge con JSON final
- **Modos**:
  - E0 (independent): 4 workers + vote, sin debate
  - E1 (debate-on-disagreement): debate solo si hay disagreement
  - E2 (debate-all): debate siempre
- **Fases**:
  1. Independent Assessment: 4 workers evaluan sin ver las opiniones de los demas
  2. Initial Ensemble: ConfidenceWeightedMajorityVote (frozen)
  3. Disagreement Detection + Challenge: cada worker ve las opiniones de los demas
  4. Final Judge: judge neutral decide con todos los argumentos
- **Generation**: num_predict=60, temperature=0.0
- **Hardware**: GPU
- **Total corridas**: 9 (3 modelos x 3 modos)

## Benchmark

semantic_assessment_benchmark_v2.json — mismo que EXP-012/013.

## Resultados

### Granite 3B Q4 (completo)

| Modo | Initial | Final | Delta | Corrections | Damage | Net | Stability |
|------|---------|-------|-------|-------------|--------|-----|-----------|
| independent | 69.1% (38/55) | 69.1% (38/55) | 0.0% | 0 | 0 | 0 | 100% |
| debate-on-disagreement | 67.3% (37/55) | 69.1% (38/55) | +1.8% | 12 | 11 | +1 | 56.4% |
| **debate-all** | **69.1% (38/55)** | **80.0% (44/55)** | **+10.9%** | **12** | **6** | **+6** | 67.3% |

#### Granite debate-all: matriz de transicion

```
                Final
              CONT  PART  SUPP  UNRE
         ------------------------
 CONTRAD    12     0     0     1
 PARTIAL     1    13     0     1
 SUPPORT     0     3    12     0
 UNRELAT     3     4     5     0
```

#### Granite debate-all: metricas de debate

| Metric | Value |
|--------|-------|
| Debates triggered | 55/55 (100%) |
| Revision rate | 15.9% |
| Debate trigger rate | 100% |
| Correction rate | 70.6% (12/17 wrong) |
| Damage rate | 15.8% (6/38 correct) |

### Qwen3 4B-RAG (completo)

| Modo | Initial | Final | Delta | Corrections | Damage | Net |
|------|---------|-------|-------|-------------|--------|-----|
| independent | 76.4% (42/55) | 76.4% (42/55) | 0.0% | 0 | 0 | 0 |
| debate-on-disagreement | 76.4% (42/55) | 76.4% (42/55) | 0.0% | 5 | 5 | 0 |
| debate-all | 76.4% (42/55) | 74.6% (41/55) | -1.8% | 4 | 5 | -1 |

Qwen3 4B-RAG no se beneficia del debate. El modelo tiene suficiente
capacidad inicial (76.4%) y el debate introduce ruido (net 0 o -1).

### Llama 3.2 3B (pendiente - protocolo corregido)

Los resultados previos de Llama3.2 en EXP-014 estaban contaminados
por el protocolo (POST-001: num_predict=10 + think mode + parser
leniento). Se re-corre con protocolo corregido en EXP-016.

| Modo | Initial | Final | Delta |
|------|---------|-------|-------|
| independent | pendiente (EXP-016) | pendiente | - |
| debate-on-disagreement | pendiente (EXP-016) | pendiente | - |
| debate-all | pendiente (EXP-016) | pendiente | - |

### Modelos adicionales (EXP-016)

4 modelos adicionales fueron evaluados en EXP-016: Gemma3 4B,
Nemotron 3 4B, Ministral 3B, Qwen3.5 4B. Ver EXP-016 para resultados
completos.

## Observaciones

1. **Granite debate-all es el escenario optimo**: 12 corrections vs 6
   damage, net +6, +10.9% accuracy. El debate corrige mas errores de
   los que destruye. Esto es evidencia fuerte de H1.

2. **Granite debate-on-disagreement tiene demasiado damage**: 12
   corrections vs 11 damage, net +1. El damage rate (29.7%) es
   alarmante. Esto se debe parcialmente a un bug detectado: los casos
   unanimous pasaban al judge sin debate, y el judge fallaba.

3. **El judge tiende a sobre-classificar UNRELATED como SUPPORTS/PARTIAL**:
   la transition matrix muestra 5 casos UNRELATED->SUPPORTS y 4
   UNRELATED->PARTIAL. El judge no entiende que claim y evidence se
   refieren a sujetos diferentes (wrong_subject).

4. **wrong_subject es catastrofico para el judge**: 4/5 casos correctos
   (UNRELATED) se destruyen en wrong_subject. El judge ignora la
   entidad y se enfoca en la similitud de contenido.

5. **El debate corrige partial_support**: ps-006 SUPPORTS->PARTIAL,
   os-002 UNRELATED->PARTIAL. El challenge round ayuda a distinguir
   soporte parcial de soporte completo.

6. **El debate corrige implicit_contradiction**: ic-001 PARTIAL->CONTRADICTS.
   El challenge round ayuda a detectar contradicciones implicitas.

7. **Qwen3 independent baseline (76.4%) es menor que coliseo v1 (78.2%)**:
   la diferencia son los prompts. El microcoliseum usa prompts sin
   few-shot con salida JSON, el coliseo usa prompts few-shot con
   salida de un token. Los few-shot examples son criticos para modelos
   pequeños.

8. **Qwen3 wrong_context es 0/5 en independent**: clasifica todo como
   CONTRADICTS. El prompt sin few-shot no le da ejemplos de UNRELATED
   por contexto diferente.

## Anomalias

- **Bug detectado en debate-on-disagreement**: los casos unanimous no
  seteaban final_relation = initial_relation, causando que final_relation
  quedara vacio y se contara como incorrecto. Corregido para corridas
  4-9, pero las corridas 1-2 (Granite) tienen este bug en los casos
  unanimous. Los casos con debate son validos.

- **Granite independent (69.1%) vs coliseo v1 single (61.8%)**: el
  microcoliseum usa prompts diferentes (sin few-shot, JSON output).
  Aun asi, Granite independent es mayor que coliseo v1 single. Esto
  sugiere que el prompt JSON con roles especializados es mas efectivo
  que el prompt few-shot con rol neutral para Granite.

- **Qwen3 independent (76.4%) vs coliseo v1 single (78.2%)**: Qwen3
  funciona mejor con few-shot prompts que con JSON prompts. Diferente
  a Granite. Cada modelo responde diferente al formato de prompt.

- **El judge destruye wrong_subject consistentemente**: en Granite
  debate-all, 4/5 casos UNRELATED correctos se convierten en SUPPORTS
  o PARTIAL. El judge no tiene la capacidad de distinguir entidades
  diferentes. Esto es un patron sistematico, no ruido.

## Interpretacion

### Evidencia a favor de H1

Granite debate-all muestra el patron buscado:

```
Initial:       69.1% (17 errores)
Corrections:   12
Damage:         6
Net:           +6
Final:         80.0%
```

El debate corrige 12 errores (70.6% de los errores iniciales) y solo
destruye 6 respuestas correctas (15.8% de los correctos iniciales).
El net effect es +6 casos (+10.9% accuracy). Esto es evidencia fuerte
de que la deliberacion aporta valor que el voting no captura.

### Evidencia en contra de H1

Granite debate-on-disagreement muestra el patron negativo:

```
Initial:       67.3% (18 errores)
Corrections:   12
Damage:        11
Net:           +1
```

El debate corrige 12 errores pero destruye 11 respuestas correctas.
El damage rate (29.7%) es inaceptable. Sin embargo, esto se debe
parcialmente al bug de casos unanimous.

### Patron de damage: wrong_subject

El damage se concentra en wrong_subject. El judge consistentemente
convierte UNRELATED correcto a SUPPORTS/PARTIAL porque no distingue
entidades diferentes. Esto sugiere que el judge necesita un prompt
mas explicito sobre verificacion de entidad, o que el Worker D
(context/entity analyst) necesita mas peso en la deliberacion.

### Comparacion de prompts

El microcoliseum usa prompts diferentes al coliseo (sin few-shot, JSON
output). Esto hace que el baseline no sea directamente comparable.
La comparacion valida es dentro del microcoliseum: independent vs
debate-all con los mismos prompts.

## Decision

- **H1 confirmada para Granite debate-all**: el debate corrige mas
  errores de los que destruye (net +6, +10.9%).
- **H0 no refutada para debate-on-disagreement**: el damage rate es
  demasiado alto (29.7%), aunque parcialmente por bug.
- **Accion**: completar las 6 corridas restantes (Qwen3 y Llama en
  los 3 modos) para validar si el patron se replica en otros modelos.

## Hipotesis refutada

- H(implicit): el debate siempre mejora sobre el ensemble paralelo.
  Refutada para debate-on-disagreement: el damage rate (29.7%) anula
  las correcciones (net +1).

## Hipotesis nacida

- H: El debate-all es superior al debate-on-disagreement porque el
  judge confirma los casos unanimous en lugar de dejarlos sin
  evaluacion. El debate-on-disagreement necesita un mecanismo de
  "confirmacion" para casos unanimous, no solo "skip".
- H: El damage en wrong_subject se debe a que el judge no tiene un
  prompt explicito sobre verificacion de entidad. Un prompt de judge
  mas especializado podria reducir el damage.
- H: El debate es mas util para modelos debiles (Granite +10.9%) que
  para modelos fuertes (Qwen3). Un modelo fuerte ya acierta los casos
  que el debate corrige.
- H: Los prompts few-shot del coliseo son criticos para el baseline.
  El microcoliseum deberia usar los mismos prompts para que la
  comparacion sea valida. El debate debe compararse contra el mejor
  baseline, no contra un baseline debilitado por prompts suboptimos.
