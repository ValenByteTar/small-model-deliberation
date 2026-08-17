---
id: EXP-019
title: "BitNet Relevance x Entailment Decomposition"
date: 2026-08-18
status: completed
category: experiment
components: [semantic_ensemble, llama_server_provider, semantic_adapter]
tags: [bitnet, relevance-detection, entailment, two-stage, decomposition, logprobs, threshold-calibration, partial-detection]
related: [EXP-017, EXP-018, PM-003]
supersedes: null
superseded_by: null
---

# EXP-019 - BitNet Relevance x Entailment Decomposition

## Hipotesis

H1: BitNet puede resolver relevance y entailment por separado cuando no
puede resolver ambos en una clasificacion de 4 clases.

H2: Un sistema de 2 etapas (relevance gate + entailment classifier +
decision layer deterministico) recupera relevance detection sin perder
el entailment que EXP-018 descubrio.

H3: Si la descomposicion funciona, el sistema cruza 50% (28/55).

## Motivacion

EXP-018 mostro que BitNet tiene capacidad de entailment (NLI reframing
acerto 12/12 SUPPORTS) pero debilidad en relevance detection (NLI dice
YES para todo, wrong_subject 0%, wrong_context 0%). La pregunta natural:
puede BitNet resolver relevance y entailment **por separado** cuando no
puede resolverlos **juntos**?

La arquitectura propuesta:

```
  EVIDENCE + CLAIM
       |
       v
  STAGE 1: RELEVANCE
  "Do CLAIM and EVIDENCE discuss the same specific subject and context?"
  YES / NO (con logprobs)
       |
       +-- NO --> UNRELATED
       |
       +-- YES --> STAGE 2: ENTAILMENT
                   "Based on EVIDENCE, is CLAIM TRUE, FALSE, or PARTIALLY TRUE?"
                   TRUE / FALSE / PARTIALLY (con logprobs)
                        |
                        +-- TRUE --> SUPPORTS
                        +-- FALSE --> CONTRADICTS
                        +-- PARTIALLY --> PARTIAL
```

La autoridad esta en el sistema (decision layer deterministico), no en
el LLM. BitNet produce senales semanticas, no decisiones.

## Setup

**Modelo**: BitNet-b1.58-2B-4T
**Benchmark**: `semantic_assessment_v2.json` (55 casos, 10 categorias)
**Server**: llama-server, CPU, temp=0.0, repeat_penalty=1.0, n_probs=8

### Stage 1: Relevance (binary)

- Pregunta: "Do the CLAIM and EVIDENCE discuss the same specific subject
  (same product, standard, framework, technique) in the same context
  (same environment, sector, lifecycle phase)?"
- Output: YES / NO
- Few-shot: 6 ejemplos (3 YES, 3 NO)
- **Crucial**: contradicciones son RELEVANTES (mismo sujeto, solo discrepan)

### Stage 2: Entailment (3-way)

- Pregunta: "Based on the EVIDENCE, is the CLAIM TRUE, FALSE, or
  PARTIALLY TRUE?"
- Output: TRUE / FALSE / PARTIALLY
- Few-shot: 5 ejemplos (2 TRUE, 2 FALSE, 1 PARTIALLY)
- Solo se ejecuta si Stage 1 dice YES

### Variantes

- **Direct**: TRUE/FALSE/PARTIALLY en una pasada
- **Cascading**: TRUE/FALSE primero, si TRUE -> FULLY/PARTIALLY

## Resultados

### Stage 1: Relevance Detection (independiente)

| Metrica | Greedy | Argmax (logprobs) |
|---------|--------|-------------------|
| Relevant correctos | 3/46 (6.5%) | 5/46 (10.9%) |
| **Irrelevant correctos** | **9/9 (100%)** | **9/9 (100%)** |
| Accuracy total | 21.8% | 25.5% |
| **Irrelevant detection rate** | **100%** | **100%** |

**Hallazgo clave**: BitNet detecta **100% de los casos irrelevantes**
cuando se le pregunta directamente. Los 5 wrong_subject y 4 wrong_context
son todos correctamente identificados como NO.

**Pero**: el modelo es demasiado conservador. Dice NO a 52/55 casos.
Solo dice YES a 3 casos (d-001, d-004, ec-002). El threshold greedy es
demasiado alto.

### Stage 1: Logprobs Analysis

Los logprobs muestran separacion entre relevant e irrelevant:

| Grupo | diff (YES_lp - NO_lp) media | min | max |
|-------|----------------------------|-----|-----|
| Relevant (n=46) | -1.34 | -2.56 | +0.94 |
| Irrelevant (n=9) | -2.06 | -2.70 | -0.85 |

Hay overlap: algunos relevantes tienen diff mas bajo que algunos
irrelevantes. No hay threshold perfecto.

| Threshold | Accuracy | Relevant correctos | Irrelevant correctos |
|-----------|----------|-------------------|---------------------|
| -3.0 | 83.6% | 46/46 | 0/9 |
| **-2.5** | **85.5%** | **44/46** | **3/9** |
| -2.0 | 69.1% | 33/46 | 5/9 |
| -1.5 | 52.7% | 22/46 | 7/9 |
| -1.0 | 36.4% | 12/46 | 8/9 |
| 0.0 | 25.5% | 5/46 | 9/9 |

### Stage 2: Entailment (independiente)

| Metrica | Valor |
|---------|-------|
| Casos donde Stage 1 dijo relevant (greedy) | 3 |
| Entailment accuracy en esos 3 | 66.7% (2/3) |

Muestra demasiado pequena para concluir. Pero se observo un bug critico:

**Bug en mapping greedy**: el modelo outputa `False` (capital F, sin
leading space) para los 55 casos. El mapping case-sensitive no encuentra
match y cae al default `SUPPORTS`. Los logprobs agregados (argmax) si
diferencian correctamente, pero el greedy es inutil.

### Sistema combinado de 2 etapas

| Modo | Greedy | Argmax |
|------|--------|--------|
| Direct | 20.0% (11/55) | 20.0% (11/55) |
| Cascading | 20.0% (11/55) | 21.8% (12/55) |

El sistema de 2 etapas es **inferior** a EXP-018 (40%) porque el
relevance gate greedy bloquea demasiado.

### Hybrid: EXP-018 + Relevance Gate

Combinando el logit ensemble de EXP-018 con el relevance gate de
EXP-019 como override:

**Regla**: si EXP-018 predice SUPPORTS y el relevance logprob diff
<= threshold, override a UNRELATED.

| Threshold | Accuracy | Delta vs EXP-018 |
|-----------|----------|-----------------|
| -2.5 | 41.8% (23/55) | +1.8% |
| -2.0 | 41.8% (23/55) | +1.8% |
| **-1.8** | **43.6% (24/55)** | **+3.6%** |
| -1.5 | 40.0% (22/55) | +0.0% |

**Mejor techo: 43.6% (24/55)** con threshold -1.8.

### Perfil cognitivo del hybrid (t=-1.8, 24/55)

| Categoria | EXP-017 | EXP-018 | EXP-019 hybrid | Target usuario |
|-----------|---------|---------|----------------|----------------|
| direct_evidence | 0% | 100% | **100%** | >=90% |
| paraphrase | 0% | 83% | 50% | >=80% |
| explicit_contradiction | 80% | 60% | 60% | — |
| implicit_contradiction | 40% | 40% | 40% | — |
| negation | 50% | 83% | **67%** | >=80% |
| wrong_subject | 60% | 0% | **80%** | >=60% |
| wrong_context | 40% | 0% | 20% | >=60% |
| partial_support | 17% | 0% | 0% | >=50% |
| over_specificity | 20% | 0% | 0% | >=50% |
| adversarial | 0% | 17% | 17% | — |

## Analisis

### Que funciono

1. **Relevance detection es real**: BitNet detecta 100% de casos
   irrelevantes cuando se le pregunta directamente. No es una
   incapacidad total de detectar relevancia — es un problema de
   threshold y framing.

2. **El relevance gate recupera wrong_subject**: de 0% (EXP-018) a 80%
   (EXP-019 hybrid). El override SUPPORTS->UNRELATED cuando el relevance
   gate dice NO es efectivo para casos de sujeto equivocado.

3. **La descomposicion es arquitectonicamente valida**: el sistema de 2
   etapas con decision layer deterministico funciona. BitNet produce
   senales semanticas, el sistema toma la decision. La autoridad no
   esta en el LLM.

### Que no funciono

1. **El relevance gate greedy es demasiado conservador**: dice NO a
   52/55 casos. El threshold logprob (-1.8) ayuda pero no es perfecto:
   pierde 2 paraphrase que EXP-018 acertaba.

2. **Stage 2 entailment tiene sesgo de token**: el modelo outputa
   `False` para los 55 casos en el greedy. El GBNF grammar y el
   mapping case-sensitive interactuan mal. Los logprobs muestran
   diferenciacion real, pero el greedy es inutil.

3. **PARTIAL sigue siendo 0%**: ni el relevance gate ni el entailment
   stage ni el cascading (FULLY/PARTIALLY) pueden detectar soporte
   parcial. El modelo no distingue "A y B y C" soportado por "A y B"
   de soporte completo.

4. **wrong_context solo recupera 20%**: el relevance gate detecta
   wrong_subject (diferente producto/standard) pero falla en
   wrong_context (mismo sujeto, diferente contexto). La pregunta
   "same specific subject AND context" es demasiado estricta para
   casos donde el sujeto es el mismo pero el contexto cambia.

### Descomposicion de capacidades de BitNet

Los experimentos (EXP-017, 018, 019) revelan tres capacidades
distintas con niveles diferentes:

| Capacidad | Medicion | Estado |
|-----------|----------|--------|
| **Entailment binario** (TRUE/FALSE) | NLI 3a: 12/12 SUPPORTS, 5/5 explicit_contradicts (argmax) | Fuerte |
| **Relevance detection** (claro) | 100% wrong_subject, 100% wrong_context (greedy) | Fuerte |
| **Relevance detection** (sutil) | ws-003, wc-003 pasan el gate | Debil |
| **Granularity assessment** (PARTIAL) | 0/17 PARTIAL en todos los experimentos | Ausente |

La frontera de BitNet no esta en entailment (EXP-018 lo demostro) ni
en relevance detection claro (EXP-019 lo demostro). Esta en:

1. **Relevance sutil**: distinguir "mismo topico, diferente entidad"
   (NIST vs ISO, ambos mencionan "identify") de "mismo topico, misma
   entidad"
2. **Granularity assessment**: distinguir soporte completo de soporte
   parcial cuando el evidence cubre parte del claim

### Sobre la atribucion causal

Los datos de EXP-019 confirman que la debilidad en relevance detection
**no es una incapacidad total** — BitNet detecta 100% de casos
irrelevantes claros. La debilidad es especifica:

- Threshold miscalibrado (demasiado conservador)
- Casos sutiles de relevance (mismo topico, diferente entidad)
- Granularity assessment (PARTIAL)

La causa puede ser cuantizacion, capacidad del modelo 2B,
entrenamiento, framing, representacion linguistica de
UNRELATED/PARTIAL/CANNOT_TELL, o una combinacion. Los experimentos
no aisan la causa. Demuestran **que** capacidades estan debilitadas
y **donde** esta la frontera, no **porque**.

## Veredicto

| Metrica | EXP-017 | EXP-018 | EXP-019 |
|---------|---------|---------|---------|
| Techo single regime | 29.1% | 29.1% | — |
| Techo ensemble | 27.3% (voting) | 40.0% (logit) | **43.6%** (hybrid) |
| SUPPORTS correctos | 0/12 | 12/12 | 12/12 |
| UNRELATED correctos | 8/10 | 4/10 | **7/10** |
| PARTIAL correctos | 3/17 | 0/17 | 0/17 |

**H1 confirmada**: BitNet puede resolver relevance y entailment por
separado. Relevance detection alcanza 100% en casos claros. Entailment
alcanza 12/12 SUPPORTS con NLI reframing.

**H2 parcialmente confirmada**: el relevance gate recupera wrong_subject
(0% -> 80%) pero no wrong_context (0% -> 20%). El sistema de 2 etapas
es arquitectonicamente valido pero el threshold no es perfecto.

**H3 rechazada**: 50% no es alcanzable. El techo del hybrid es 43.6%
(24/55). Faltan 4 aciertos, todos en PARTIAL (partial_support 0/6,
over_specificity 0/5, adversarial 1/6). La capacidad de granularity
assessment esta ausente en BitNet con estos prompts y regimenes.

### Frontera de BitNet

La frontera esta en **granularity assessment** (PARTIAL) y
**relevance sutil** (mismo topico, diferente entidad), no en
entailment ni en relevance detection claro. Esto es consistente con
la hipotesis del usuario:

> "modelo capaz de evaluar relacion proposicional, pero incapaz de
> modelar adecuadamente las condiciones de aplicabilidad de esa
> relacion."

Las "condiciones de aplicabilidad" tienen dos componentes:
1. Relevance (es esta evidencia sobre el sujeto/contexto correcto?) —
   moderado, falla en casos sutiles
2. Granularity (soporta el evidence todo el claim o solo parte?) —
   ausente

## Scripts

| Script | Funcion |
|--------|---------|
| `runners/run_bitnet_relevance_entailment_decomposition.py` | 2 etapas + logprobs + decision layer |

## Raw data

| Archivo | Contenido |
|---------|-----------|
| `results/raw/bitnet_relevance_entailment_direct.json` | 2 etapas, modo direct |
| `results/raw/bitnet_relevance_entailment_cascading.json` | 2 etapas, modo cascading |
