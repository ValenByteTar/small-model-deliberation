---
id: EXP-017
title: "Barrido multidimensional y perfil cognitivo de BitNet b1.58-2B-4T"
date: 2026-08-17
status: completed
category: experiment
components: [semantic_ensemble, llama_server_provider, semantic_adapter, debate]
tags: [bitnet, cognitive-profile, multidimensional-sweep, gbnf, constrained-generation, json-schema, few-shot, ensemble, decoding, temperature, repeat-penalty, binary-cascading, cpu]
related: [EXP-015, EXP-016, PM-003, POST-001]
supersedes: null
superseded_by: null
---

# EXP-017 - Barrido multidimensional y perfil cognitivo de BitNet

## Hipotesis

H1: Existe algun regimen de inferencia (carga cognitiva minima, GBNF
constrained decoding, few-shot calibrado, esquemas jerarquicos, decoding
alternativo) en el que BitNet b1.58-2B-4T conserve capacidad semantica
util o especializada que lo haga viable como evaluador semantico.

H2: El ensemble de multiples workers puede corregir errores individuales
y elevar la accuracy por encima del techo del worker individual.

H3: BitNet tiene un perfil cognitivo especifico (fortaleza en algunas
categorias, debilidad en otras) que podria aprovecharse arquitectonicamente.

## Motivacion

PM-003 documento que BitNet no superaba el 50% accuracy en evaluacion
semantica. EXP-015 confirmo 29.1% single en Coliseo v1 con protocolo
corregido. EXP-016 descubrio que el microcoliseum producía artefactos
(BitNet ecoaba el template JSON, dando todo CONTRADICTS por orden
alfabetico del parser).

Antes de descartar definitivamente BitNet, era necesario aislar
sistematicamente todas las dimensiones que podrian afectar su rendimiento:
prompt, decoding, formato de salida, carga cognitiva, y deliberacion.

Este experimento responde a la pregunta: **¿hay algun regimen donde
BitNet pase de mediocre a sorprendentemente competitivo?**

## Configuracion

- **Modelo**: BitNet-b1.58-2B-4T (i2_s ternario, 1.58 bits por peso)
- **Backend**: llama-server (bitnet.cpp), CPU, 4 threads
- **Benchmark**: semantic_assessment_v2.json (55 casos, 10 categorias)
- **Hardware**: CPU (sin GPU)

### Dimensiones exploradas

#### Dimension 1: Carga Cognitiva y Formato de Salida

| Régimen | Descripción |
|---------|-------------|
| Régimen 0 (Zero-Shot GBNF) | `CLAIM + EVIDENCE -> GBNF 1-token`, sin few-shot |
| Régimen 1 (Few-Shot GBNF) | 4 ejemplos balanceados (1 por clase) + GBNF 1-token |
| Régimen 2 (Binary Cascading) | Paso 1: `¿Relevante? YES/NO` -> Paso 2: `SUPPORTS/CONTRADICTS/PARTIAL` |

#### Dimension 2: Decoding e Hiperparámetros

| Parámetro | Valores explorados |
|-----------|-------------------|
| temperature | 0.0 (determinismo) vs 0.2 (evitar minimos locales) |
| top_k | 1 (greedy) vs 20 (muestreo restringido) |
| repeat_penalty | 1.0 (sin penalizacion) vs 1.1 (penalizacion estandar) |

#### Dimension 3: Ensemble y Deliberacion

| Modo | Descripción |
|------|-------------|
| single | 1 worker (neutral) |
| ensemble_2 | 2 workers (entailment, skeptical) + majority vote |
| ensemble_4 | 4 workers (entailment, skeptical, contradiction, neutral) + majority vote |

## Hallazgos tecnicos preliminares

### Hallazgo 1: BitNet ecoa templates JSON (EXP-016)

**Problema**: BitNet no entiende la instruccion de rellenar un template
JSON con valores reales. Cuando ve:
```
{"worker": "A", "relation": "SUPPORTS|PARTIAL|CONTRADICTS|UNRELATED", ...}
```
copia el template verbatim, incluyendo el string literal con todas las
opciones. El parser `_normalize_relation` encuentra "CONTRADICTS"
primero (alfabetico) -> todo da CONTRADICTS -> 27.3% es artefacto.

**Solucion**: `json_schema` por request en llama-server activa
constrained generation (GBNF interno) que fuerza JSON valido. Pero el
JSON completo satura la capacidad del modelo (2B ternario no puede
coordinar semantica + sintaxis JSON + rol + confidence simultaneamente).

**Solucion optima**: GBNF grammar directo a 1 token (solo la palabra
de la relacion), sin JSON, sin explicacion, sin confidence. Minima
carga cognitiva.

### Hallazgo 2: repeat_penalty=1.1 distorsiona BitNet

**Problema**: `repeat_penalty=1.1` penaliza tokens que aparecen en el
prompt. Las 4 relaciones aparecen en el prompt ("Relations: SUPPORTS,
PARTIAL, CONTRADICTS, UNRELATED"). La penalizacion es proporcional al
logit original: BitNet tiene tendencia natural hacia SUPPORTS (logit
mas alto), repeat_penalty lo penaliza mas fuertemente, empujando al
siguiente token (PARTIAL).

**Evidencia** (mismo prompt, mismo schema, variando repeat_penalty):

| Caso | Expected | repeat_penalty=1.1 | repeat_penalty=1.0 |
|------|----------|-------------------|-------------------|
| d-001 | SUPPORTS | PARTIAL (incorrecto) | SUPPORTS (correcto) |
| ec-001 | CONTRADICTS | SUPPORTS (incorrecto) | SUPPORTS (incorrecto) |
| ws-001 | UNRELATED | PARTIAL (incorrecto) | SUPPORTS (incorrecto) |
| ps-001 | PARTIAL | PARTIAL (correcto) | PARTIAL (correcto) |

Con repeat_penalty=1.1: 1/4 correcto, sesgo a PARTIAL
Sin repeat_penalty: 2/4 correcto, sesgo a SUPPORTS

**Solucion**: repeat_penalty=1.0 estricto para BitNet.

### Hallazgo 3: Chain-of-Thought empeora BitNet

**Problema**: Pedirle a BitNet que genere una explicacion (`REASON:`)
antes de la clasificacion empeora el resultado. El modelo copia
fragmentos de los ejemplos few-shot ("The evidence states that the
framework does not exist...") y los inserta como razon, derivando en
CONTRADICTS.

**Evidencia**:

| Configuracion | Accuracy (5 casos) |
|--------------|-------------------|
| Few-Shot + GBNF (solo palabra) | 40% (2/5) |
| Few-Shot + CoT guiado (REASON + RELATION) | 40% (2/5) |
| Few-Shot + CoT + definiciones + 8-shot | 25% (2/8) |

El CoT no ayuda porque BitNet alucina negaciones ("is not required")
que no existen en la evidencia.

**Solucion**: Sin CoT, sin explicacion, salida directa a 1 token.

## Resultados

### Barrido multidimensional completo (7 configuraciones)

| # | Régimen / Configuracion | Accuracy | Correctos | Wall |
|---|------------------------|----------|-----------|------|
| 1 | Régimen 0: Zero-Shot GBNF (temp=0.0) | 25.4% | 14/55 | 0.5m |
| 2 | Régimen 0: Zero-Shot GBNF (temp=0.2, top_k=20) | 21.8% | 12/55 | 0.5m |
| 3 | Régimen 1: Few-Shot GBNF (temp=0.0, rep=1.0) | **29.1%** | **16/55** | 0.5m |
| 4 | Régimen 1: Few-Shot GBNF (temp=0.2, top_k=20) | 25.4% | 14/55 | 0.6m |
| 5 | Régimen 2: Binary Cascading / One-vs-All | 27.3% | 15/55 | 1.0m |
| 6 | Ensemble_2 (Few-Shot GBNF, voting) | 27.3% | 15/55 | 1.1m |
| 7 | Ensemble_4 (Few-Shot GBNF, voting) | 27.3% | 15/55 | 2.0m |

**Mejor régimen**: #3 (Few-Shot GBNF, temp=0.0, repeat_penalty=1.0,
single worker) con 29.1%.

### Análisis por dimensión

#### Dimension 1: Carga Cognitiva

- **Zero-Shot (Régimen 0)**: 25.4%. Sin ejemplos, el modelo colapsa
  a CONTRADICTS (no entiende la tarea).
- **Few-Shot (Régimen 1)**: 29.1%. Los 4 ejemplos balanceados le dan
  el patron a seguir. Es el mejor regimen.
- **Binary Cascading (Régimen 2)**: 27.3%. La pregunta binaria de
  relevancia funciona (detecta UNRELATED), pero la segunda fase
  sobre-clasifica en PARTIAL.

#### Dimension 2: Decoding

- **temperature=0.0**: 29.1% (mejor). Determinismo puro.
- **temperature=0.2 + top_k=20**: 25.4%. El muestreo con temperatura
  introduce ruido sin beneficiar la diversidad. BitNet no tiene
  suficiente capacidad para que la diversidad de sampling produzca
  respuestas mejores, solo mas aleatorias.
- **repeat_penalty=1.1**: destruye la distribucion (demostrado en
  hallazgo 2). repeat_penalty=1.0 es obligatorio.

#### Dimension 3: Ensemble

- **single**: 29.1% (mejor).
- **ensemble_2**: 27.3%. El voting no corrige errores porque los
  workers tienen alta correlacion (votan igual en la mayoria de los
  casos). En algunos casos el desempate empeora.
- **ensemble_4**: 27.3%. Misma correlacion de errores. Confirmacion
  directa de PM-003 (Jaccard 0.40-0.64).

**El ensemble no ayuda — empeora ligeramente.** Esto refuta H2.

### Perfil cognitivo por categoria (mejor régimen: single Few-Shot GBNF)

| Categoria | Casos | BitNet Accuracy | Prediccion dominante |
|-----------|-------|----------------|---------------------|
| explicit_contradiction | 5 | **80.0%** (4/5) | CONTRADICTS (4) |
| wrong_subject | 5 | **60.0%** (3/5) | UNRELATED (3) |
| negation | 6 | 50.0% (3/6) | CONTRADICTS (2), PARTIAL (2), UNRELATED (2) |
| implicit_contradiction | 5 | 40.0% (2/5) | CONTRADICTS (2), UNRELATED (3) |
| wrong_context | 5 | 40.0% (2/5) | CONTRADICTS (2), UNRELATED (2) |
| over_specificity | 5 | 20.0% (1/5) | CONTRADICTS (2), UNRELATED (2) |
| partial_support | 6 | 16.7% (1/6) | CONTRADICTS (3), UNRELATED (2) |
| direct_evidence | 6 | **0.0%** (0/6) | CONTRADICTS (2), UNRELATED (3) |
| paraphrase | 6 | **0.0%** (0/6) | UNRELATED (5) |
| adversarial | 6 | **0.0%** (0/6) | UNRELATED (3), CONTRADICTS (3) |
| **TOTAL** | **55** | **29.1%** (16/55) | — |

### Matriz comparativa: BitNet 2B (optimo) vs Qwen3.5 4B (debate-all)

```
                             BitNet 2B (Opt)       Qwen3.5 4B          Delta
──────────────────────────────────────────────────────────────────────────────
direct_evidence (6)             0/6 (  0.0%)          6/6 (100.0%)      +100.0%
paraphrase (6)                  0/6 (  0.0%)          6/6 (100.0%)      +100.0%
partial_support (6)             1/6 ( 16.7%)          4/6 ( 66.7%)       +50.0%
explicit_contradiction (5)      4/5 ( 80.0%)          5/5 (100.0%)       +20.0%
implicit_contradiction (5)      2/5 ( 40.0%)          5/5 (100.0%)       +60.0%
negation (6)                    3/6 ( 50.0%)          6/6 (100.0%)       +50.0%
over_specificity (5)            1/5 ( 20.0%)          4/5 ( 80.0%)       +60.0%
wrong_subject (5)               3/5 ( 60.0%)          5/5 (100.0%)       +40.0%
wrong_context (5)               2/5 ( 40.0%)          3/5 ( 60.0%)       +20.0%
adversarial (6)                 0/6 (  0.0%)          5/6 ( 83.3%)       +83.3%
──────────────────────────────────────────────────────────────────────────────
TOTAL GLOBAL (55)              16/55 ( 29.1%)        49/55 ( 89.1%)      +60.0%
```

## Observaciones

### 1. BitNet tiene un perfil cognitivo bipartito

**Fortaleza residual**: deteccion de contradicciones explicitas (80%) y
negaciones (50%). El modelo puede detectar cuando un claim y un evidence
dicen cosas incompatibles a nivel literal (numeros diferentes, hechos
opuestos).

**Debilidad catastrofica**: entailment y parafasis (0%). El modelo no
entiende que dos textos con vocabulario diferente pueden significar lo
mismo. "Credential dumping can expose authentication material" y
"Attackers may extract reusable secrets from operating-system credential
stores" son semanticamente equivalentes, pero BitNet los clasifica como
UNRELATED porque no comparten suficientes tokens.

### 2. La cuantizacion destruye exactamente lo que se necesita

La reduccion a 1.58 bits por peso destruye la capacidad de representar
matices semanticos finos. El modelo retiene:
- Deteccion de conflicto literal (contradicciones explicitas)
- Deteccion de diferencia de topico (wrong_subject)

Pero pierde:
- Reconocimiento de sinonimos y parafasis
- Inferencia de soporte logico (entailment)
- Deteccion de sutilezas contextuales

Esto es consistente con la teoria: la cuantizacion extrema preserva
patrones lexicales gruesos pero destruye las representaciones distribuidas
finas necesarias para entailment.

### 3. Ningun regimen supera el 29.1%

Se exploraron 7 configuraciones distintas cubriendo:
- 3 regimenes de carga cognitiva (zero-shot, few-shot, binary cascading)
- 2 configuraciones de decoding (greedy, sampling con temperatura)
- 3 configuraciones de ensemble (single, 2-workers, 4-workers)

Ninguna supero el 29.1%. El techo del modelo esta acotado por su
capacidad de representacion, no por el regimen de inferencia.

### 4. El ensemble confirma correlacion de errores

Los 4 workers votan igual en la mayoria de los casos (alta correlacion).
El voting no corrige errores porque los workers comparten el mismo sesgo.
Esto confirma PM-003 (Jaccard 0.40-0.64) y refuta H2.

### 5. Chain-of-Thought es contraproducente

BitNet alucina negaciones ("is not required", "does not exist") que no
estan en la evidencia. El CoT no ayuda porque el modelo no tiene capacidad
de razonamiento logico suficiente; solo repite fragmentos de los ejemplos.

### 6. repeat_penalty es un artefacto para modelos pequeños

`repeat_penalty=1.1` (estandar en muchos runners) distorsiona la
distribucion de logits en prompts estructurados donde las opciones
aparecen en el prompt. Para modelos pequeños con vocabulario de 4
opciones, esto empuja al modelo hacia el segundo token mas probable,
produciendo sesgos sistematicos (PARTIAL en este caso).

## Conclusion

### H1: Refutada

No existe ningun regimen de inferencia donde BitNet conserve capacidad
semantica util para evaluacion semantica. El mejor regimen (Few-Shot
GBNF, temp=0.0, repeat_penalty=1.0, single worker) alcanza 29.1%, con
0% en entailment y parafasis.

### H2: Refutada

El ensemble no mejora sobre el worker individual (27.3% vs 29.1%).
Los workers tienen alta correlacion de errores.

### H3: Parcialmente confirmada pero irrelevante

BitNet tiene un perfil cognitivo especifico (80% en explicit_contradiction,
60% en wrong_subject), pero Qwen3.5 alcanza 100% en esas mismas
categorias. La fortaleza residual de BitNet es redundante: cualquier
modelo de 4B la supera.

### Veredicto final

**La reduccion de precision a 1.58 bits destruye justamente la capacidad
de entailment que este componente necesita.** BitNet b1.58-2B-4T no es
viable como evaluador semantico, confirmado de forma exhaustiva y
multidimensional.

El modelo de referencia indiscutido para evaluacion semantica local es
**Qwen3.5 4B (89.1% con deliberacion, 0 damage)**.

## Scripts creados

- `runners/run_bitnet_cognitive_profiling.py`: Perfil cognitivo
  (single, ensemble_2, ensemble_4) con regimen optimo.
- `runners/run_bitnet_multidimensional_sweep.py`: Barrido factorial
  de 4 regimenes (zero-shot, zero-shot+temp, few-shot+temp, binary
  cascading).

## Datos crudos

- `results/raw/bitnet_cognitive_profile_single.json`
- `results/raw/bitnet_cognitive_profile_ensemble_2.json`
- `results/raw/bitnet_cognitive_profile_ensemble_4.json`
- `results/raw/bitnet_multidimensional_sweep.json`
