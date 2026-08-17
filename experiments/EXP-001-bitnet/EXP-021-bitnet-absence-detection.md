---
id: EXP-021
title: "BitNet Absence Detection Falsation Probe"
date: 2026-08-18
status: completed
category: experiment
components: [semantic_ensemble, llama_server_provider, semantic_adapter]
tags: [bitnet, absence-detection, falsation, keyword-matching, compositional, final-verdict]
related: [EXP-017, EXP-018, EXP-019, EXP-020, PM-003]
supersedes: null
superseded_by: null
---

# EXP-021 - BitNet Absence Detection Falsation Probe

## Hipotesis

H0 (nula): BitNet puede usar ausencia de evidencia como condicion
negativa para una inferencia composicional.

H1 (alternativa): BitNet no puede usar ausencia de evidencia como
condicion negativa. Dice TRUE sistematicamente para claims no-soportados.

## Motivacion

EXP-020 mostro que el comportamiento observable de BitNet es
consistente con keyword matching holistico: dice TRUE
para cualquier proposicion que comparta keywords con el evidence, sin
verificar ausencia. Atomic FALSE accuracy: 0%.

EXP-021 es el experimento de falsacion final. Tres condiciones
minimas, una sola pregunta, una sola metrica. Sin framing complejo,
sin decomposition, sin aggregation. Solo: puede BitNet decir FALSE
cuando debe?

## Setup

**Modelo**: BitNet-b1.58-2B-4T
**Benchmark**: `absence_detection_v1.json` (18 casos, 3 condiciones)
**Server**: llama-server, CPU, temp=0.0, repeat_penalty=1.0, n_probs=8
**Few-shot**: 2 ejemplos (1 TRUE, 1 FALSE) con instruccion explicita:
"Answer FALSE if any part of the claim is not confirmed by the evidence."

### Tres condiciones

| Condicion | Evidence | Claim | Expected | Que mide |
|-----------|----------|-------|----------|----------|
| implicit_absence | A + B | A + B + C | FALSE | Ausencia implicita de C |
| explicit_negation | A + B | A + B + NOT-C | TRUE | Negacion explicita consistente |
| total_absence | A + B | C | FALSE | Ausencia total de C |

6 casos por condicion (3 dominios: encryption, protocol, tool x 2
repeticiones con dominios diferentes).

### Metrica

- greedy TRUE/FALSE
- logprob(TRUE), logprob(FALSE)
- margen = logP(TRUE) - logP(FALSE)

## Resultados

### Tabla completa

| ID | Condicion | Expected | Greedy | lp(TRUE) | lp(FALSE) | Margen | OK? |
|----|-----------|----------|--------|----------|-----------|--------|-----|
| ia-001 | implicit_absence | FALSE | TRUE | -0.268 | -1.921 | +1.653 | X |
| ia-002 | implicit_absence | FALSE | TRUE | -0.635 | -1.001 | +0.365 | X |
| ia-003 | implicit_absence | FALSE | TRUE | -0.559 | -1.117 | +0.558 | X |
| ia-004 | implicit_absence | FALSE | TRUE | -0.607 | -1.050 | +0.443 | X |
| ia-005 | implicit_absence | FALSE | TRUE | -0.470 | -1.292 | +0.822 | X |
| ia-006 | implicit_absence | FALSE | TRUE | -0.348 | -1.659 | +1.312 | X |
| en-001 | explicit_negation | TRUE | TRUE | -0.408 | -1.473 | +1.066 | OK |
| en-002 | explicit_negation | TRUE | TRUE | -0.704 | -0.921 | +0.217 | OK |
| en-003 | explicit_negation | TRUE | TRUE | -0.784 | -0.845 | +0.061 | OK |
| en-004 | explicit_negation | TRUE | TRUE | -0.721 | -0.951 | +0.229 | OK |
| en-005 | explicit_negation | TRUE | TRUE | -0.598 | -1.087 | +0.489 | OK |
| en-006 | explicit_negation | TRUE | TRUE | -0.460 | -1.345 | +0.885 | OK |
| ta-001 | total_absence | FALSE | TRUE | -0.504 | -1.190 | +0.686 | X |
| ta-002 | total_absence | FALSE | TRUE | -0.497 | -1.274 | +0.777 | X |
| ta-003 | total_absence | FALSE | TRUE | -0.499 | -1.300 | +0.800 | X |
| ta-004 | total_absence | FALSE | TRUE | -0.563 | -1.191 | +0.628 | X |
| ta-005 | total_absence | FALSE | TRUE | -0.555 | -1.165 | +0.611 | X |
| ta-006 | total_absence | FALSE | TRUE | -0.540 | -1.196 | +0.656 | X |

### Resumen por condicion

| Condicion | Casos | Correctos | Accuracy | TRUE | FALSE | Margen medio | Margen min |
|-----------|-------|-----------|----------|------|-------|--------------|------------|
| explicit_negation | 6 | 6 | 100.0% | 6 | 0 | +0.491 | +0.061 |
| implicit_absence | 6 | 0 | 0.0% | 6 | 0 | +0.859 | +0.365 |
| total_absence | 6 | 0 | 0.0% | 6 | 0 | +0.693 | +0.611 |
| **TOTAL** | **18** | **6** | **33.3%** | | | | |

### Analisis de falsacion

**Casos donde expected=FALSE (12 casos):**
- Correctos (greedy=FALSE): **0/12 (0.0%)**
- Incorrectos (greedy=TRUE): **12/12 (100.0%)**
- Margen logP(TRUE)-logP(FALSE):
  - medio: **+0.776**
  - min: **+0.365**
  - max: **+1.653**
- Casos con margen > 0 (TRUE > FALSE): **12/12**

**Casos donde expected=TRUE (6 casos, explicit_negation):**
- Correctos (greedy=TRUE): 6/6 (100.0%)
- Margen medio: +0.491

## Veredicto

**H0 rechazada. H1 confirmada.**

BitNet dice TRUE sistematicamente para claims no-soportados:
- 0/12 FALSE correctos
- 12/12 TRUE incorrectos
- Margen siempre positivo (TRUE > FALSE en los 12 casos)
- Margen minimo: +0.365 (confianza alta incluso en el peor caso)

**BitNet no usa FALSE como senal de ausencia bajo el framing evaluado
en este experimento.** Esto es diferente de "BitNet no puede emitir
FALSE": EXP-023 demuestra que BitNet emite False/FALSE naturalmente en
53/55 casos bajo grammar permisivo o sin grammar. El hallazgo
correcto es que **el token FALSE no aparece como respuesta a la
ausencia de evidencia bajo este framing particular** (NLI 3a con
grammar estricto, que fuerza TRUE). Queda pendiente determinar si
esto es una incapacidad semantica o un artefacto del grammar (EXP-024).

### Observacion: explicit_negation funciona

La unica condicion donde BitNet acerto (6/6) es explicit_negation:
cuando el claim dice "does not support C" y el evidence no menciona C,
BitNet dice TRUE. Pero esto no es verificacion de ausencia — es
matching positivo por overlap: el claim y el evidence comparten A y B, y
BitNet dice TRUE por overlap parcial. El "NOT-C" no es verificado, es
ignorado.

Si el claim dijera "does not support A" (negando algo presente en el
evidence), BitNet probablemente diria TRUE tambien, por el mismo
mecanismo de matching por overlap. La condicion explicit_negation no
distingue verificacion de ausencia de matching positivo.

### Por que el few-shot no ayuda

El few-shot incluye la instruccion explicita: "Answer FALSE if any
part of the claim is not confirmed by the evidence." Y un ejemplo
donde la respuesta correcta es FALSE (claim A+B+C, evidence A+B).

BitNet ignora la instruccion y el ejemplo. Su mecanismo de inference
es consistente con matching holistico: si hay suficiente overlap de keywords
entre claim y evidence, dice TRUE. La ausencia de C no es detectada
porque C no es buscado como ausente — es ignorado.

## Conclusion final de la cadena experimental

La cadena EXP-017 -> 018 -> 019 -> 020 -> 021 aisla progresivamente
la frontera de BitNet:

| EXP | Descubrimiento | Techo |
|-----|----------------|-------|
| EXP-017 | BitNet falla entailment y relevance con labels artificiales | 29.1% |
| EXP-018 | NLI reframing rompe SUPPORTS wall. Logit ensemble | 40.0% |
| EXP-019 | Relevance detection funciona para casos claros. Hybrid | 43.6% |
| EXP-020 | Keyword matching parcial. Atomic decomposition no funciona | 33.3% (atomic) |
| EXP-021 | **Ausencia de evidencia no es condicion negativa. 0% FALSE** | **Cierre** |

### Mapa de capacidades final

```
                         BitNet
                           |
            +--------------+--------------+
            |              |              |
         PUEDE        DEPENDE        NO PUEDE
            |              |              |
    +-------+-------+      |        +-----+---------+
    |               |      |        |               |
 presencia/     entailment relevance  ausencia    granularidad
 relevance        simple    sutil    negativa   composicional
    |               |      |        |               |
   [OK]           [OK]   debil/     [FAIL]         [FAIL]
                         inestable
```

La categoria "depende del caso" cubre relevance sutil: distinguir
"NIST vs ISO" (mismo topico, diferente entidad) es conceptualmente
distinto de "Product A vs Product B" (entidades claramente diferentes).
BitNet puede hacer lo segundo pero no lo primero de forma confiable.

### BitNet como componente

BitNet no es viable como evaluador semantico generalista. Pero no es
un fracaso total. Es un resultado arquitectonicamente util: convierte
un modelo aparentemente mediocre en un componente con una frontera
operacional bien definida.

**Capacidades confirmadas:**
- Relevance detection claro: 100% (EXP-019)
- Entailment binario (TRUE/FALSE con NLI reframing): 12/12 SUPPORTS (EXP-018)

**Capacidades ausentes:**
- Verificacion de ausencia: 0% (EXP-021)
- Granularity assessment (PARTIAL): 0% (EXP-017-020)
- Relevance detection sutil: debil (EXP-019)

### La pregunta cambia

De: "Como hacemos que BitNet sea un buen juez?"

A: "Donde es economicamente optimo utilizar la senal que BitNet si
puede producir?"

BitNet como **extractor barato de senales semanticas elementales**
(relevance claro, entailment binario), con la autoridad en el sistema
(contratos y politicas deterministas). No como evaluador semantico
completo.

### Sobre la atribucion causal

Los experimentos demuestran **que** comportamiento esta ausente (FALSE
no aparece como respuesta a ausencia bajo este framing), no **por que**.
La causa puede ser cuantizacion, capacidad del modelo 2B, entrenamiento,
framing, grammar, o combinacion. **EXP-023 introduce una variable de
confusion**: este experimento uso grammar estricto, que fuerza TRUE. Si
se replicara con grammar permisivo (bajo el cual BitNet emite FALSE
naturalmente), el resultado podria ser diferente. Queda pendiente
controlar esta variable (EXP-024).

## Scripts

| Script | Funcion |
|--------|---------|
| `runners/run_bitnet_absence_detection.py` | Falsation probe (3 condiciones) |

## Raw data

| Archivo | Contenido |
|---------|-----------|
| `results/raw/bitnet_absence_detection.json` | 18 casos + logprobs + margenes |
| `benchmarks/absence_detection_v1.json` | Benchmark controlado (18 casos) |
