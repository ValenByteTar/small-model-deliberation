---
id: EXP-018
title: "NLI Reframing + Logit Ensemble: BitNet b1.58-2B-4T"
date: 2026-08-18
status: completed
category: experiment
components: [semantic_ensemble, llama_server_provider, semantic_adapter]
tags: [bitnet, nli-reframing, logit-ensemble, logprobs, gbnf, few-shot, ensemble, decoding, support-wall, calibration]
related: [EXP-017, PM-003, POST-001]
supersedes: null
superseded_by: null
---

# EXP-018 - NLI Reframing + Logit Ensemble para BitNet

## Hipotesis

H1: El sesgo de BitNet contra el token "SUPPORTS" (0/12 en EXP-017) es un
sesgo de **token-etiqueta**, no de capacidad subyacente. Reformulando la
tarea a NLI (TRUE/FALSE/CANNOT_TELL), tokens naturales que el modelo
maneja con confianza, se rompe la SUPPORTS wall.

H2: Los logprobs del primer token (n_probs, nunca medidos en EXP-017)
revelan masa de probabilidad oculta bajo el greedy. Un logit ensemble
que sume logprobs across regimenes captura complementariedad que el
voting destruye.

H3: La combinacion de NLI reframing + logit ensemble eleva BitNet por
encima del techo de 29.1% de EXP-017, con target 50% (28/55).

## Motivacion

EXP-017 establecio un techo de 29.1% (16/55) para BitNet en
`semantic_assessment_v2.json`. El hallazgo mas revelador fue que en
**12/12 casos SUPPORTS, ninguno de los 5 regimenes emitio SUPPORTS**.
Todos usaban el token "SUPPORTS" como etiqueta de salida.

Sin embargo, el oráculo de 5 regimenes (elegir el régimen correcto a
posteriori) alcanzaba 52.7% (29/55). Hay complementariedad real:
bin cascade acierta 6/6 partial_support, fs0 acierta 4/5
explicit_contradiction. El voting (27.3%) no la captura porque majority
washes out.

Este experimento explora dos palancas no tocadas:

1. **NLI reframing**: cambiar el framing de la tarea para usar tokens
   naturales (TRUE/FALSE/CANNOT_TELL, YES/NO/PARTIALLY/NOT_MENTIONED)
   en lugar de etiquetas artificiales (SUPPORTS/CONTRADICTS/UNRELATED/PARTIAL).

2. **Logprobs + logit ensemble**: usar n_probs=4 del llama-server para
   capturar la distribucion de probabilidad del primer token, y agregar
   logprobs across regimenes con logsumexp en lugar de majority voting.

## Setup

**Modelo**: BitNet-b1.58-2B-4T (ggml-model-i2_s.gguf)
**Benchmark**: `semantic_assessment_v2.json` (55 casos, 10 categorias)
**Server**: llama-server, CPU, 4 threads, 2GB context, temp=0.0, repeat_penalty=1.0
**Decoding**: GBNF constrained + n_probs=8 para capturar top-8 tokens

### Regimenes NLI (Fase 1)

| Régimen | Framing | Etiquetas | Mapa |
|---------|---------|-----------|------|
| 3a: NLI 3-way | "Based on the evidence, the claim is:" | TRUE/FALSE/CANNOT_TELL | TRUE→SUPPORTS, FALSE→CONTRADICTS, CANNOT_TELL→UNRELATED |
| 3b: NLI Cascading | 3a + si TRUE, "fully or partially?" | FULLY/PARTIALLY | FULLY→SUPPORTS, PARTIALLY→PARTIAL |
| 3c: NLI 4-way | "Is the claim supported by the evidence?" | YES/NO/PARTIALLY/NOT_MENTIONED | YES→SUPPORTS, NO→CONTRADICTS, PARTIALLY→PARTIAL, NOT_MENTIONED→UNRELATED |

Todos con few-shot (3-4 ejemplos con vocabulario de cybersecurity) y
GBNF grammar constraint.

### Logit Ensemble (Fase 3)

Agrega 3 regimenes con logsumexp:

1. **fs0**: Few-Shot GBNF (SUPPORTS/CONTRADICTS/UNRELATED/PARTIAL)
2. **nli4**: NLI 4-way (YES/NO/PARTIALLY/NOT_MENTIONED) con grammar fix
   que permite variantes de espacio/case
3. **bin**: Binary Cascading (Relevancia YES/NO → Polaridad SUPPORTS/CONTRADICTS/PARTIAL)

Para cada caso y cada régimen, se extraen los top-8 logprobs del primer
token, se mapean a etiquetas canónicas, y se agregan con logsumexp.

## Resultados

### Fase 1: NLI Reframing (3 regimenes, greedy)

| Régimen | Accuracy | SUPPORTS correctos |
|---------|----------|-------------------|
| 3a: NLI 3-way (TRUE/FALSE/CANNOT_TELL) | 21.8% (12/55) | 12/12 |
| 3b: NLI Cascading (TRUE → FULLY/PARTIALLY) | 21.8% (12/55) | 12/12 |
| **3c: NLI 4-way (YES/NO/PARTIALLY/NOT_MENTIONED)** | **29.1% (16/55)** | **12/12** |

**H1 confirmada**: NLI reframing rompe la SUPPORTS wall. Los 3 regimenes
NLI aciertan 12/12 casos SUPPORTS (direct_evidence + paraphrase), algo
que ninguno de los 5 regimenes de EXP-017 logro.

**Pero**: el NLI 4-way (3c) tiene sesgo hacia YES. Predice SUPPORTS en
43/55 casos. Los 12 SUPPORTS correctos se compensan con 27 SUPPORTS
incorrectos. El accuracy greedy no supera el techo de EXP-017.

### Fase 2: Logprobs Diagnostic

Los logprobs del primer token revelan informacion que el greedy oculta:

| Caso | Esperado | Greedy NLI 3a | Top-1 logprob | Top-2 logprob |
|------|----------|---------------|---------------|---------------|
| ec-001 | CONTRADICTS | TRUE (SUPPORTS) | `False` -1.20 | `True` -1.82 |
| ec-002 | CONTRADICTS | TRUE (SUPPORTS) | `True` -0.72 | `TRUE` -1.43 |
| ic-001 | CONTRADICTS | TRUE (SUPPORTS) | `Cannot` -1.82 | `False` -1.86 |
| n-001 | CONTRADICTS | TRUE (SUPPORTS) | `False` -1.11 | `True` -1.79 |

**Hallazgo clave**: En casos de contradiccion explicita (ec-001, n-001),
el token `False` tiene mayor logprob que `True`, pero el GBNF fuerza
`TRUE` como greedy. El modelo **si distingue** contradiccion de soporte
en los logprobs, pero el grammar constraint selecciona el token equivocado.

NLI 3a con argmax sobre logprobs agregados (sin greedy forzado):
- explicit_contradiction: **5/5** (vs 0/5 greedy)
- implicit_contradiction: **5/5** (vs 0/5 greedy)
- direct_evidence: 0/6 (vs 6/6 greedy)
- paraphrase: 0/6 (vs 6/6 greedy)

**Complementariedad perfecta**: NLI 3a greedy acierta SUPPORTS, NLI 3a
logprob-argmax acierta CONTRADICTS. Ninguno acierta ambos.

### Fase 3: Logit Ensemble (3 regimenes, logsumexp)

| Configuracion | Accuracy | vs EXP-017 |
|--------------|----------|------------|
| fs0 single (EXP-017 baseline) | 29.1% (16/55) | — |
| nli4 single (EXP-018) | 29.1% (16/55) | — |
| bin cascade single (EXP-017) | 27.3% (15/55) | — |
| **Logit ensemble (fs0 + nli4 + bin, logsumexp)** | **38.2% (21/55)** | **+9.1%** |

### Fase 3b: Logit Ensemble con 4 regimenes + calibracion

Agregando NLI 3a como 4to régimen y calibrando pesos:

| Configuracion | Accuracy |
|--------------|----------|
| 4-regimen (fs0 + nli4 + bin + nli3a, pesos uniformes) | 36.4% (20/55) |
| **4-regimen calibrado (nli3a=3.0, nli4=1.0, bin=1.0, fs0=1.0)** | **40.0% (22/55)** |

**Mejor configuracion**: nli3a_w=3.0, nli4_w=1.0, bin_w=1.0, fs0_w=1.0

### Perfil cognitivo del logit ensemble (mejor config, 22/55)

| Categoria | EXP-017 (fs0) | EXP-018 (logit ensemble) | Delta |
|-----------|---------------|-------------------------|-------|
| direct_evidence | 0.0% | **100.0%** | +100.0% |
| paraphrase | 0.0% | **83.3%** | +83.3% |
| explicit_contradiction | 80.0% | 60.0% | -20.0% |
| implicit_contradiction | 40.0% | 40.0% | 0.0% |
| negation | 50.0% | **83.3%** | +33.3% |
| partial_support | 16.7% | 0.0% | -16.7% |
| over_specificity | 20.0% | 0.0% | -20.0% |
| wrong_subject | 60.0% | 0.0% | -60.0% |
| wrong_context | 40.0% | 0.0% | -40.0% |
| adversarial | 0.0% | 16.7% | +16.7% |
| **TOTAL** | **29.1%** | **40.0%** | **+10.9%** |

## Analisis

### Que funciono

1. **NLI reframing rompe la SUPPORTS wall**: 12/12 SUPPORTS correctos
   en los 3 regimenes NLI. El sesgo era de token, no de capacidad
   subyacente. BitNet puede reconocer soporte directo cuando se le pide
   que diga "TRUE" en lugar de "SUPPORTS".

2. **Logprobs revelan informacion oculta**: el greedy NLI 3a dice TRUE
   en casos de contradiccion, pero el logprob de `False` es mayor que
   el de `True` en ec-001 y n-001. El modelo distingue, el grammar no.

3. **Logit ensemble captura complementariedad**: fs0 detecta
   contradicciones, NLI detecta soporte, bin detecta partial. El
   logsumexp preserva la confianza de cada régimen. El voting la
   destruia.

### Que no funciono

1. **Target 50% no alcanzado**: el techo del logit ensemble calibrado
   es 40.0% (22/55). Faltan 6 aciertos para 50%.

2. **NLI 4-way contamina con sesgo YES**: predice SUPPORTS en 43/55
   casos. Su contribucion al ensemble eleva SUPPORTS pero destruye
   UNRELATED y PARTIAL.

3. **Logprobs no resuelven UNRELATED ni PARTIAL**: los regimenes NLI
   no tienen un token natural para "irrelevante" que el modelo use con
   confianza. CANNOT_TELL y NOT_MENTIONED son tokens raros en el
   vocabulario del modelo. wrong_subject cae de 60% a 0%.

4. **Calibracion de pesos es overfitting**: los pesos optimos
   (nli3a=3.0, nli4=1.0) fueron encontrados por grid search sobre el
   benchmark. No hay garantia de generalizacion.

### Por que 50% no es alcanzable con BitNet

El logit ensemble recupera SUPPORTS (12/12) y mejora CONTRADICTS
(negation 83.3%), pero **destruye UNRELATED y PARTIAL**:

- wrong_subject: 60% → 0% (los regimenes NLI dicen YES/TRUE para todo)
- wrong_context: 40% → 0% (mismo problema)
- over_specificity: 20% → 0% (no hay token natural para "parcialmente")
- partial_support: 16.7% → 0% (bin cascade acertaba 6/6, pero el
  ensemble lo domina con SUPPORTS del NLI)

El trade-off observado: **NLI reframing gana entailment pero pierde
relevancia detection**. Con estos prompts, estos regimenes y este
mecanismo de decodificacion, no hay framing unico que recupere ambas
capacidades simultaneamente.

Nota sobre atribucion causal: los datos de EXP-018 demuestran que
BitNet tiene una capacidad muy debil para detectar irrelevancia bajo
NLI reframing. No demuestran que la cuantizacion 1.58 bits sea
necesariamente la causa. Podrian existir otras explicaciones:
capacidad insuficiente del modelo 2B, entrenamiento/instruction
tuning, framing de la tarea, representacion linguistica de
UNRELATED/PARTIAL/CANNOT_TELL, dificultad de distinguir truth
evaluation de evidence relevance, o una combinacion de todas.

## Veredicto

| Metrica | EXP-017 | EXP-018 | Delta |
|---------|---------|---------|-------|
| Techo single regime | 29.1% | 29.1% | 0.0% |
| Techo ensemble | 27.3% (voting) | **40.0%** (logit) | **+12.7%** |
| SUPPORTS correctos | 0/12 | **12/12** | +12 |
| UNRELATED correctos | 8/10 | 4/10 | -4 |
| PARTIAL correctos | 3/17 | 0/17 | -3 |

**H1 — reformulada tras EXP-023**: Originalmente se interpreto como
"el sesgo contra SUPPORTS era de token, no de capacidad; NLI reframing
lo rompe." **EXP-023 demuestra que esta interpretacion es demasiado
fuerte.** El regimen NLI 3a uso grammar estricto, que fuerza el token
`TRUE` independientemente de la semantica del caso (TRUE x55 en
EXP-023). Los 12/12 SUPPORTS son consistentes con "TRUE aparece de
forma universal bajo grammar estricto y coincide con los casos
SUPPORTS", no con "BitNet discrimina entailment."

Formulacion corregida: **"Respuesta TRUE bajo grammar estricto: 12/12
en casos SUPPORTS, pero sin evidencia suficiente de discriminacion
semantica; EXP-023 demuestra que el mismo regimen produce TRUE
practicamente de forma universal."**

**H2 confirmada**: los logprobs revelan informacion que el greedy oculta.
El logit ensemble captura complementariedad que el voting destruye.
(Esta conclusion se sostiene independientemente del grammar: los
logprobs contienen informacion que el argmax destruye.)

**H3 rechazada**: 50% no es alcanzable. El techo del logit ensemble
calibrado es 40.0% (22/55). El NLI reframing gana entailment pero
pierde relevancia detection. El trade-off es fundamental para un
modelo 2B ternario.

### Implicacion para PM-003

PM-003 se sostiene. La afirmacion "BitNet tiene capacidad de
entailment" **debe retirarse** tras EXP-023: los 12/12 SUPPORTS son
consistentes con una respuesta TRUE universal inducida por grammar
estricto, no con discriminacion semantica. Lo que se observa
debilitado es:

1. La capacidad de usar etiquetas artificiales como tokens de salida
2. La capacidad de distinguir relevancia de soporte (NLI dice YES para todo)
3. La capacidad de emitir PARTIAL con confianza
4. **La capacidad de discriminar semanticamente** (pendiente de
   confirmacion en EXP-024: el grammar confunde la interpretacion)

La causa de estas debilidades no esta aislada: puede ser cuantizacion,
capacidad del modelo 2B, entrenamiento, framing, grammar, o una
combinacion. Los experimentos demuestran el **que** (capacidades
debilitadas), no el **porque** (causa raiz).

**BitNet no es viable como evaluador semantico** (40% < 60% minimo).
La razon no es solo "entailment destruido" ni un "trade-off entre
entailment y relevancia": es que **no hemos podido separar la
capacidad semantica del sesgo de decodificacion inducido por
grammar.** EXP-024 buscara aislar estas variables.

### Caveat: Grammar como variable de primer orden (EXP-023)

**EXP-023 demostro que el grammar GBNF es una variable experimental
de primer orden para BitNet.** Este experimento uso grammars
**estrictos** (`root ::= "TRUE" | "FALSE" | "CANNOT_TELL"`, sin
espacios ni variantes de caso), lo que fuerza a BitNet a generar
`TRUE` para todo (55/55 casos). Con grammars permisivos (con
espacios), BitNet genera `FALSE` para todo (53/55 casos).

La "SUPPORTS wall" que NLI reframing rompio fue parcialmente un
artefact del grammar estricto, no solo del token labeling. Si este
experimento hubiera usado grammars permisivos, el resultado habria
sido una "CONTRADICTS wall" en lugar de una "SUPPORTS wall".

Las conclusiones sobre logit ensemble y complementariedad siguen
siendo validas, pero la atribucion causal ("el sesgo era de token,
no de capacidad") debe matizarse: **el grammar tambien contribuye al
sesgo observado.** Ver EXP-023 para detalles.

## Scripts

| Script | Funcion |
|--------|---------|
| `runners/run_bitnet_nli_reframing.py` | Fase 1+2: 3 regimenes NLI + logprobs |
| `runners/run_bitnet_logit_ensemble.py` | Fase 3: logit ensemble (fs0 + nli4 + bin) |

## Raw data

| Archivo | Contenido |
|---------|-----------|
| `results/raw/bitnet_nli_regimen_3a.json` | NLI 3-way con logprobs |
| `results/raw/bitnet_nli_regimen_3b.json` | NLI Cascading con logprobs |
| `results/raw/bitnet_nli_regimen_3c.json` | NLI 4-way con logprobs |
| `results/raw/bitnet_nli_reframing_all.json` | Los 3 regimenes juntos |
| `results/raw/bitnet_logit_ensemble.json` | Logit ensemble (3 regimenes) |
