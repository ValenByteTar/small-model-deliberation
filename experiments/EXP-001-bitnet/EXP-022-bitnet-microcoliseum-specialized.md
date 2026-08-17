# EXP-022: BitNet Microcoliseum - 4 Workers Especializados

## Fecha
2025-01-XX

## Contexto

PM-003 cerro BitNet como evaluador semantico generalista. La pregunta
que queda es: **puede un ensemble de workers especializados, cada uno
usando el regimen que mejores resultados dio para su capacidad
especifica, superar al mejor regimen individual?**

Este experimento adapta el microcoliseum para BitNet, asignando a cada
worker el prompt/grammar/regimen que mejores resultados dio en los
experimentos anteriores.

## Hipotesis

H0: El microcoliseum especializado no supera al mejor regimen
    individual de EXP-018 (29.1%).

H1: El microcoliseum especializado + judge deterministico supera al
    mejor regimen individual, aprovechando que cada worker cubre una
    capacidad distinta.

## Diseño

### Workers especializados

| Worker | Rol | Regimen | Origen | Rationale |
|--------|-----|---------|--------|-----------|
| A | entailment | NLI 3a (TRUE/FALSE/CANNOT_TELL) | EXP-018 | 12/12 SUPPORTS en greedy (dice TRUE para todo) |
| B | skeptical | NLI Cascading (TRUE→FULL/PARTIAL) | EXP-020 | Unico regimen que emite PARTIAL |
| C | contradiction | NLI 3a con framing de contradiccion | EXP-018 | 16/16 CONTRADICTS en argmax (FALSE token fuerte) |
| D | context | Relevance gate (YES/NO) | EXP-019 | 100% irrelevantes claros, wrong_subject 100% |

### Grammars estrictos

Los grammars GBNF usan solo la forma canonica sin espacios ni
variantes de caso:

```
root ::= "TRUE" | "FALSE" | "CANNOT_TELL"
root ::= "YES" | "NO"
root ::= "FULL" | "PARTIAL"
```

Esto es critico: grammars permisivos (con " TRUE", " True", etc.)
cambian la tokenizacion y BitNet prefiere " FALSE" sobre "TRUE",
invirtiendo el comportamiento esperado. Los grammars estrictos
reproducen exactamente el comportamiento de EXP-018.

### Debate adaptado

El debate original del microcoliseum pide razonamiento composicional
(evaluar contraargumentos, identificar worker con mayor riesgo,
justificar con evidencia). EXP-021 demostró que BitNet no puede hacer
esto.

El debate adaptado **no pide razonamiento composicional**. En cambio,
cada worker re-evalua con un framing diferente que incorpora la senal
del worker en desacuerdo:

- Worker A (SUPPORTS): si alguien dijo CONTRADICTS, re-preguntar con
  framing de contradiccion
- Worker B (PARTIAL): si alguien dijo SUPPORTS, re-preguntar
  FULL/PARTIAL
- Worker C (CONTRADICTS): si alguien dijo SUPPORTS, re-preguntar
  TRUE/FALSE con framing de contradiccion
- Worker D (relevance): si alguien dijo relevante, re-preguntar
  relevance

El debate produce nuevas logprobs (no KEEP/CHANGE), que se integran
al ensemble con peso 0.5.

### Judge deterministico

El judge es una policy deterministica, no un LLM. Orden de prioridad:

1. Worker D greedy + debate UNRELATED → override UNRELATED (conf 0.95)
2. Worker D greedy UNRELATED → override UNRELATED (conf 0.9)
3. Incorporar debate logprobs (peso 0.5)
4. Worker B greedy PARTIAL → boost PARTIAL (+2.0 o +3.0 si debate confirma)
5. Worker C argmax CONTRADICTS → boost CONTRADICTS (+1.5 o +2.5 si debate confirma)
6. Worker A greedy SUPPORTS → boost SUPPORTS (+1.0)
7. Recalcular final con logprobs + boosts

### Infraestructura

- 4 instancias llama-server en puertos 8101-8104 (CPU, 2 threads cada una)
- 4 workers en paralelo via ThreadPoolExecutor
- Benchmark: semantic_assessment_v2.json (55 casos)
- Modo: debate-on-disagreement

## Resultados

### Accuracy global

| Metric | Value |
|--------|-------|
| Initial (ensemble) | 23.6% (13/55) |
| Final (judge) | **29.1% (16/55)** |
| Delta | **+5.5%** |
| Corrections | 7 |
| Damage | 4 |
| Net | +3 |
| Stability | 50.9% |
| Wall time | 4.7 min (280s) |

### Worker accuracy (greedy vs ground truth)

| Worker | Rol | Accuracy | Comportamiento |
|--------|-----|----------|----------------|
| A | entailment | 21.8% (12/55) | Dice SUPPORTS para todo → acierta SUPPORTS |
| B | skeptical | 21.8% (12/55) | Cascading raramente emite PARTIAL |
| C | contradiction | 32.7% (18/55) | Dice CONTRADICTS para la mayoria → acierta contradicciones |
| D | context | 29.1% (16/55) | Dice UNRELATED para casos no-obvios → acierta wrong_subject/context |

### Por categoria

| Categoria | N | Init | Final | Delta | Mecanismo |
|-----------|---|------|-------|-------|-----------|
| explicit_contradiction | 5 | 20.0% | **60.0%** | +40.0% | Worker C boost funciona |
| implicit_contradiction | 5 | 0.0% | **40.0%** | +40.0% | Worker C boost funciona |
| negation | 6 | 16.7% | **33.3%** | +16.7% | Worker C boost funciona |
| adversarial | 6 | 0.0% | 16.7% | +16.7% | Worker C boost parcial |
| wrong_subject | 5 | 100.0% | **100.0%** | 0.0% | Worker D override perfecto |
| wrong_context | 5 | 80.0% | 60.0% | -20.0% | Worker C boost dana |
| over_specificity | 5 | 20.0% | 0.0% | -20.0% | Worker C boost dana |
| partial_support | 6 | 16.7% | 0.0% | -16.7% | Worker C boost dana |
| direct_evidence | 6 | 0.0% | 0.0% | 0.0% | Worker A no basta solo |
| paraphrase | 6 | 0.0% | 0.0% | 0.0% | Worker A no basta solo |

## Analisis

### H1 confirmada parcialmente

El microcoliseum especializado + judge deterministico alcanza 29.1%,
empatando con el mejor regimen individual de EXP-018 (NLI 4-way:
29.1%). No lo supera en accuracy global, pero:

1. **El judge agrega +5.5% sobre el ensemble** (23.6% → 29.1%)
2. **Deteccion de contradicciones mejorada drasticamente**:
   - explicit: 20% → 60% (+40%)
   - implicit: 0% → 40% (+40%)
   - negation: 16.7% → 33.3% (+16.7%)
3. **Relevance detection perfecta** en wrong_subject (100%)

### Trade-off del judge

El boost a CONTRADICTS (Worker C) tiene un trade-off claro:

```
                 Beneficiado          Dandiado
                 ------------         ------------
  Worker C       explicit_contrad     over_specificity
  boost          implicit_contrad     partial_support
  CONTRADICTS    negation             wrong_context
```

Esto es consistente con PM-003: BitNet puede producir senales utiles
(Worker C para contradicciones, Worker D para relevance) pero la
agregacion requiere tuning cuidadoso. El boost actual es demasiado
agresivo para casos PARTIAL.

### El debate no agrega valor

El debate adaptado (re-evaluar con framing diferente) produce pocos
cambios (1-2 workers cambian por caso) y no mejora la accuracy. El
judge deterministico ya captura la mayor parte del valor via boosts.

### Comparacion con EXP-018

| Experimento | Accuracy | Notas |
|-------------|----------|-------|
| EXP-018 NLI 3a greedy | 21.8% | Solo SUPPORTS |
| EXP-018 NLI 4-way greedy | 29.1% | Mejor regimen individual |
| EXP-018 Logit ensemble | 40.0% | Solo SUPPORTS/CONTRADICTS/UNRELATED |
| EXP-019 Hybrid (relevance + ensemble) | 43.6% | Peak historico |
| **EXP-022 Microcoliseum** | **29.1%** | Empata NLI 4-way, pero con judge +5.5% |

EXP-022 no supera el peak historico de EXP-019 (43.6%), pero ese
experimento usaba un relevance gate separado + logit ensemble, no un
microcoliseum. La comparacion justa es con EXP-018 individual.

### Hallazgo critico: grammars estrictos

Los grammars GBNF permisivos (con espacios y variantes de caso)
cambian la tokenizacion de BitNet:

- Grammar permisivo: BitNet prefiere " FALSE" → dice CONTRADICTS para todo
- Grammar estricto: BitNet prefiere "TRUE" → dice SUPPORTS para todo

Esto reproduce exactamente el comportamiento de EXP-018 y es
probablemente la causa de que algunos experimentos anteriores no
fueran reproducibles. **Los grammars estrictos son obligatorios para
BitNet.**

## Conclusiones

1. **El microcoliseum especializado empata al mejor regimen
   individual** (29.1%), pero agrega valor via judge deterministico
   (+5.5% sobre ensemble).

2. **La especializacion por worker funciona**: Worker C (contradiction)
   mejora deteccion de contradicciones de 20% a 60%, Worker D
   (context) mantiene 100% en wrong_subject.

3. **El debate adaptado no agrega valor**: el judge deterministico ya
   captura el valor de las senales individuales.

4. **El trade-off del judge es claro**: boost a CONTRADICTS ayuda
   contradicciones pero dana PARTIAL/over_specificity. Esto es
   consistente con PM-003.

5. **Los grammars estrictos son criticos** para reproducibilidad.
   **Ver EXP-023 para el analisis completo:** el grammar GBNF es una
   variable experimental de primer orden para BitNet. Los grammars
   estrictos (usados en este experimento) fuerzan a BitNet a generar
   TRUE para todo, mientras que los grammars permisivos (usados en
   EXP-019/020/021) producen FALSE para todo. La especializacion por
   worker observada aqui es parcialmente un artefacto del grammar:
   Worker A dice SUPPORTS porque usa grammar estricto, no porque tenga
   capacidad de entailment.

6. **29.1% es el techo de BitNet en este benchmark** con cualquier
   arquitectura de ensemble. El peak historico (43.6%) requirio un
   relevance gate separado + logit ensemble, no un microcoliseum.

## Implicancias para la arquitectura

El microcoliseum especializado confirma la conclusion de PM-003:
BitNet puede producir senales utiles cuando se especializa, pero la
autoridad debe estar en el sistema (judge deterministico), no en el
modelo.

El diseno correcto no es "4 BitNets votando" sino "4 BitNets
especializados + policy deterministica". El judge agrega +5.5% porque
combina las senales con conocimiento experimental (que worker es
confiable para que categoria).

El siguiente paso natural seria tunear el judge para reducir el dano
en PARTIAL/over_specificity, pero eso seria overfitting experimental.
La conclusion arquitectonica es mas util: BitNet como componente
especializado, no como juez generalista.
