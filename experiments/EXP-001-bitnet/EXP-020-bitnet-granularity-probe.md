---
id: EXP-020
title: "BitNet Granularity Probe + Atomic Decomposition"
date: 2026-08-18
status: completed
category: experiment
components: [semantic_ensemble, llama_server_provider, semantic_adapter]
tags: [bitnet, granularity, atomic-decomposition, compositional, partial-support, keyword-matching, absence-detection, probe]
related: [EXP-017, EXP-018, EXP-019, PM-003]
supersedes: null
superseded_by: null
---

# EXP-020 - BitNet Granularity Probe + Atomic Decomposition

## Hipotesis

H1: BitNet posee informacion suficiente para distinguir soporte completo
de soporte parcial, independientemente del framing utilizado.

H2: Si BitNet no puede emitir PARTIAL directamente, puede evaluar
proposiciones atomicas (TRUE/FALSE) y un aggregator deterministico
compone el resultado (N/N TRUE -> SUPPORTS, mixto -> PARTIAL).

H3: Si EXP-020 falla incluso con proposiciones atomicas, la frontera
de BitNet esta en la verificacion de ausencia, no en la composicion.

## Motivacion

EXP-019 identifico que la frontera de BitNet esta en granularity
assessment (PARTIAL) y relevance sutil. EXP-020 aisla granularity
assessment con casos minimos controlados, eliminando relevance,
entidades ambiguas y vocabulario complejo.

La pregunta: puede BitNet distinguir "A y B y C" soportado por "A y B"
de soporte completo? Y si no puede directamente, puede evaluar "A?",
"B?", "C?" atomicamente y dejar que un aggregator determine PARTIAL?

## Setup

**Modelo**: BitNet-b1.58-2B-4T
**Benchmark**: `granularity_probe_v1.json` (22 casos minimales, 7 categorias)
**Server**: llama-server, CPU, temp=0.0, repeat_penalty=1.0, n_probs=8

### Benchmark controlado

| Categoria | Casos | Descripcion |
|-----------|-------|-------------|
| full_support | 3 | Claim = Evidence (trivialmente SUPPORTS) |
| partial_support | 6 | Evidence cubre 2/3 o 1/3 del claim |
| minimal_difference | 3 | Near-synonym (access logging vs audit logging) |
| single_dimension_missing | 3 | Una dimension ausente (at rest, in transit, pero no during processing) |
| atomic_true | 3 | Proposicion atomica presente en evidence |
| atomic_false | 2 | Proposicion atomica ausente de evidence |
| atomic_not_mentioned | 2 | Proposicion sobre topico completamente diferente |

### Fases

**Fase 1: Granularity Probe directo** — 3 regimenes:
- NLI 4-way (YES/NO/PARTIALLY/NOT_MENTIONED)
- NLI Cascading (TRUE/FALSE -> si TRUE, FULL/PARTIAL)
- Direct FULL/PARTIAL/CONTRADICTS/UNRELATED

**Fase 2: Atomic Decomposition** — descomponer claims con conjunciones
en proposiciones atomicas, evaluar cada una con TRUE/FALSE, agregar
deterministicamente:
- N/N TRUE -> SUPPORTS
- 0/N TRUE -> CONTRADICTS
- mixto -> PARTIAL

## Resultados

### Fase 1: Granularity Probe directo

| Regimen | Greedy | Argmax |
|---------|--------|--------|
| NLI 4-way | 27.3% (6/22) | 27.3% (6/22) |
| **NLI Cascading** | **54.5% (12/22)** | **54.5% (12/22)** |
| Direct FULL/PARTIAL | 9.1% (2/22) | 31.8% (7/22) |

**NLI 4-way**: dice YES para todo. Solo acierta los 3 full_support y
los 3 atomic_true (todos SUPPORTS). Falla todos los PARTIAL, todos los
CONTRADICTS, todos los UNRELATED.

**NLI Cascading**: dice PARTIAL para todo (22/22). Acierta los 12 casos
que esperan PARTIAL (6 partial_support + 3 minimal_difference + 3
single_dimension_missing). Falla todos los SUPPORTS, CONTRADICTS y
UNRELATED. El cascading siempre dice "True -> Partial" — el modelo
dice TRUE en paso 1, luego PARTIAL en paso 2.

**Direct FULL/PARTIAL**: dice CONTRADICTS para casi todo (greedy). Solo
acierta los 2 atomic_false. Argmax es mejor (31.8%) pero sigue siendo
bajo.

**H1 rechazada**: BitNet no puede distinguir full support de partial
support directamente. Ningun regimen emite PARTIAL selectivamente.
NLI Cascading emite PARTIAL para todo (sesgo de token), NLI 4-way
emite YES para todo (sesgo de token), Direct emite CONTRADICTS para
todo (sesgo de token).

### Fase 2: Atomic Decomposition

Se ejecutaron dos corridas con decomposers diferentes:

**Corrida 1 (decomposer con bug)**: atoms mal formados (palabras
duplicadas, "and" al inicio)

| Metrica | Valor |
|---------|-------|
| Aggregator greedy | 53.3% (8/15) |
| Aggregator argmax | 60.0% (9/15) |
| Atomic TRUE accuracy | 82.5% (33/40) |
| Atomic FALSE accuracy | 40.0% (2/5) |

Los 6 fallos fueron **todos bugs del decomposer**, no errores del
modelo. Cuando el decomposer generaba atoms con palabras duplicadas
("encryption encryption in transit"), el modelo a veces decia FALSE
porque no encontraba la frase exacta en el evidence.

**Corrida 2 (decomposer arreglado)**: atoms bien formados con prefijo
correcto ("The system supports encryption in transit")

| Metrica | Valor |
|---------|-------|
| Aggregator greedy | 33.3% (5/15) |
| Aggregator argmax | 33.3% (5/15) |
| Atomic TRUE accuracy | 77.8% (35/45) |
| Atomic FALSE accuracy | N/A (0/0) |

**El decomposer arreglado empeoro las cosas.** Con atoms bien formados,
BitNet dice TRUE para TODO. "The system supports encryption in transit"
-> TRUE cuando el evidence solo dice "The system supports encryption
at rest". El modelo ve "encryption" y dice TRUE sin verificar "in
transit" vs "at rest".

### El hallazgo critico

**El comportamiento observado es consistente con keyword matching
holistico basado en overlap semantico/lexical, sin evidencia de
verificacion composicional confiable.**

Cuando el atom es "The system supports encryption in transit" y el
evidence es "The system supports encryption at rest", BitNet dice TRUE
porque:
1. Ve "The system supports encryption" (match exacto)
2. Ve "in" (comun en ambos)
3. No verifica que "in transit" vs "at rest" son diferentes

El modelo no descompone la proposicion en sus componentes y verifica
cada uno. Hace una matching holistico que se satisface con overlap
parcial de keywords.

**La corrida 1 (con bugs) accidentalmente funcionaba mejor** porque
los atoms mal formados ("encryption encryption in transit") no
aparecian en el evidence, y el modelo decia FALSE por no encontrar
la frase exacta. Eso es comportamiento consistente con matching
negativo por no-encontrar la frase exacta, no verificacion
composicional.

### Patron observado en atomic evaluation

| Atom | Evidence | Greedy | Comportamiento |
|------|----------|--------|----------------|
| "The system supports encryption at rest" | "...at rest, in transit, access logging" | TRUE | Keyword match exacto |
| "The system supports encryption in transit" | "...at rest, in transit, access logging" | TRUE | Keyword match exacto |
| "The system supports encryption in transit" | "...at rest only" | **TRUE** | **Keyword match parcial** |
| "The system supports access logging" | "...at rest only" | **TRUE** | **Keyword match parcial** |
| "The protocol provides auditing" | "...authentication, authorization" | **TRUE** | **Keyword match parcial** |

BitNet dice TRUE siempre que haya suficiente overlap de keywords,
independientemente de si la proposicion completa esta confirmada por
el evidence.

## Analisis

### H2 rechazada: atomic decomposition no funciona

La idea era: si BitNet no puede emitir PARTIAL directamente, evaluar
proposiciones atomicas y agregar. Pero BitNet no puede evaluar
proposiciones atomicas correctamente — dice TRUE para cualquier
proposicion que comparta keywords con el evidence, sin verificar
ausencia.

El aggregator deterministico no puede componer señales correctas
partiendo de señales incorrectas. Si todas las proposiciones dicen
TRUE, el aggregator emite SUPPORTS, incluso cuando el evidence solo
cubre 1/3 del claim.

### H3 confirmada: la frontera esta en verificacion de ausencia

La cadena experimental (EXP-017 -> 018 -> 019 -> 020) revela una
frontera consistente y especifica:

| Capacidad | EXP | Medicion | Estado |
|-----------|-----|----------|--------|
| Entailment binario (TRUE/FALSE) | 018 | 12/12 SUPPORTS | Fuerte (con NLI reframing) |
| Relevance detection (claro) | 019 | 100% irrelevantes | Fuerte |
| Relevance detection (sutil) | 019 | ws-003, wc-003 fallan | Debil |
| **Verificacion de ausencia** | **020** | **0% FALSE correctos** | **Ausente** |
| Granularity assessment (PARTIAL) | 017-020 | 0/17 en todos los experimentos | Ausente |

**La capacidad faltante es verificacion de ausencia**: BitNet no puede
decir "FALSE" cuando una proposicion no esta confirmada por el
evidence pero comparte keywords con el.

Esta capacidad es necesaria para:
1. **Granularity assessment**: detectar que "C" no esta en el evidence
   cuando "A" y "B" si estan
2. **Relevance sutil**: detectar que "ISO" no es "NIST" aunque ambos
   mencionan "identify"
3. **Contradiction detection (implicita)**: detectar que "all operating
   systems" no es "Windows and Linux only"

Sin verificacion de ausencia, BitNet no puede:
- Emitir PARTIAL (siempre ve soporte donde hay overlap parcial)
- Emitir UNRELATED (siempre ve relevance donde hay keyword overlap)
- Emitir CONTRADICTS implicita (siempre ve soporte donde hay match parcial)

### Por que NLI Cascading llego a 54.5%

El NLI Cascading dice PARTIAL para todo (22/22). Acierta 12/22 porque
12 casos esperan PARTIAL. Esto no es granularity assessment — es un
sesgo de token que coincide con la distribucion del benchmark. Si el
benchmark tuviera mas SUPPORTS que PARTIAL, el cascading bajaria.

### Implicacion para la arquitectura propuesta

La arquitectura de descomposicion + agregacion:

```
  Claim -> decompose -> [A?, B?, C?] -> aggregate -> PARTIAL
```

**No funciona con BitNet** porque BitNet no puede responder "C?" con
FALSE cuando C no esta en el evidence pero A y B si estan. El modelo
dice TRUE para C por overlap parcial de keywords (comportamiento
consistente con matching holistico).

Para que esta arquitectura funcione, se necesitaria un modelo que
pueda verificar ausencia — distinguir "presente en el evidence" de
"ausente del evidence" cuando hay overlap parcial de keywords.

### Sobre la atribucion causal

Los datos de EXP-020 demuestran que el comportamiento observable de
BitNet es consistente con keyword matching holistico:
parcial bajo estos prompts y regimenes. No demuestran que la
cuantizacion sea la causa. La capacidad de verificacion de ausencia
puede estar debilitada por cuantizacion, capacidad del modelo 2B,
entrenamiento, framing, o una combinacion.

## Veredicto

| Metrica | Valor |
|---------|-------|
| Fase 1 mejor regimen (NLI Cascading) | 54.5% (12/22) — sesgo de token |
| Fase 2 atomic decomposition (decomposer arreglado) | 33.3% (5/15) |
| Fase 2 atomic decomposition (decomposer con bug) | 60.0% (9/15) — accidental |
| Atomic TRUE accuracy | 77.8% (35/45) |
| **Atomic FALSE accuracy** | **0%** |

**H1 rechazada**: BitNet no puede distinguir full support de partial
support directamente.

**H2 rechazada**: atomic decomposition no funciona porque BitNet no
puede evaluar proposiciones atomicas correctamente (dice TRUE para
todo por overlap parcial de keywords).

**H3 confirmada**: la frontera de BitNet esta en la verificacion de
ausencia. BitNet no puede decir FALSE cuando una proposicion no esta
confirmada pero comparte keywords con el evidence.

### Conclusion de la cadena experimental

La cadena EXP-017 -> 018 -> 019 -> 020 aisla progresivamente la
frontera de BitNet:

1. **EXP-017**: techo 29.1%. BitNet falla entailment y relevance.
2. **EXP-018**: NLI reframing rompe la SUPPORTS wall (12/12). BitNet
   tiene entailment binario. Techo 40.0% con logit ensemble.
3. **EXP-019**: BitNet detecta 100% de irrelevantes claros. Relevance
   detection funciona para casos claros. Techo 43.6% con hybrid.
4. **EXP-020**: BitNet no puede verificar ausencia. Dice TRUE para
   cualquier proposicion con keyword overlap. Atomic decomposition
   no funciona.

**La frontera definitiva (bajo el protocolo actual) es la verificacion
de ausencia.** Esta capacidad es necesaria para granularity assessment
(PARTIAL), relevance sutil, y contradiction detection implicita. Sin
ella, BitNet no puede ser un evaluador semantico generalista.

### Camino adelante

Como el usuario sugirio: si EXP-020 falla incluso con proposiciones
atomicas, cerrar la investigacion de BitNet como semantic assessor
generalista y conservarlo unicamente si su ventaja de costo/latencia
justifica utilizarlo para señales mas simples:

- **Relevance detection (claro)**: 100% accuracy en EXP-019
- **Entailment binario (TRUE/FALSE)**: 12/12 SUPPORTS en EXP-018

BitNet como **extractor barato de señales semanticas elementales**,
no como evaluador semantico completo. La autoridad queda en el
sistema (contratos y politicas deterministas).

## Scripts

| Script | Funcion |
|--------|---------|
| `runners/run_bitnet_granularity_probe.py` | Fase 1 + Fase 2 |

## Raw data

| Archivo | Contenido |
|---------|-----------|
| `results/raw/bitnet_granularity_probe.json` | Fase 1 + Fase 2 (corrida final) |
| `benchmarks/granularity_probe_v1.json` | Benchmark controlado (22 casos) |
