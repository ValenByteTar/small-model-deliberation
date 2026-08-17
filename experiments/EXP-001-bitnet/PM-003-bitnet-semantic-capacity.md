---
id: PM-003
category: postmortem
status: resolved
created: 2026-08-16
updated: 2026-08-18
author: human
components: [llm_support, bitnet_provider, semantic_ensemble, semantic_adapter, kernel]
tags: [bitnet, llm-support, semantic-evaluation, ensemble, capacity-limit, deprecation, experiment-failure, nli-reframing, logit-ensemble]
related: [ADR-0031, RES-004, RES-016, EXP-010, EXP-015, EXP-017, EXP-018, PM-002, POST-001]
supersedes: null
superseded_by: null
---

# PM-003 - BitNet-b1.58-2B-4T como Semantic Capability Provider: capacidad insuficiente

## Incident

ADR-0031 (aceptado 2026-08-15) autorizo LLMSupport Fase 1 como observador
paralelo pasivo usando BitNet-b1.58-2B-4T en CPU. La hipotesis original
(RES-004) era que un modelo 2B ternario podria producir hipotesis utiles
observando el runtime sin bloquear el pipeline.

Se ejecutaron tres experimentos progresivos:

1. **Pilot 1 (hipothesis generation)**: BitNet como generador de hipotesis
   GOOD_EVIDENCE / RETRY_RETRIEVAL. Resultado: 0% de utilidad. El modelo
   no seguia el formato few-shot. Produjo 45 RETRY_RETRIEVAL sobre 50 casos
   (90%) sin razonamiento coherente.

2. **Pilot 2 (semantic assessment)**: BitNet como Semantic Capability
   Provider. Tarea replanteada: clasificar relacion claim-evidence como
   SUPPORTS / CONTRADICTS / UNRELATED / PARTIAL. Resultado: 33.3% accuracy
   (4/12), 100% adherencia al protocolo. El modelo sigue el formato pero
   no tiene capacidad semantica suficiente.

3. **Pilot 3 (ensemble de 4 workers)**: 4 instancias de BitNet con prompts
   deliberadamente diferentes (entailment, skeptical, contradiction,
   neutral). Resultado: best single 50%, best ensemble 41.7%. El ensemble
   no supera al mejor worker individual. Alta correlacion de errores
   (Jaccard 0.40-0.64).

**Impacto**: 3 sesiones de experimentacion, ~6 horas de compute, 5.5GB RAM
para el ensemble. Ningun beneficio al pipeline. ADR-0031 queda deprecado.

## Root Cause

La causa raiz **no es** un bug de integracion, ni un problema de prompt,
ni un problema de compilacion de BitNet. La causa raiz es:

**BitNet-b1.58-2B-4T no tiene capacidad semantica suficiente para
evaluar relaciones claim-evidence.**

Evidencia que confirma esta conclusion:

1. **Adherencia al protocolo 100%**: el modelo SI sigue instrucciones de
   formato. Si el problema fuera el prompt o la integracion, el protocolo
   fallaria. No falla.

2. **Accuracy 33% (azar con 4 clases = 25%)**: el modelo apenas supera
   azar. No hay comprension semantica de la relacion claim-evidence.

3. **Reasoning incoherente**: el modelo produce texto que no refleja
   comprension del par claim-evidence. Ej: "CLAIM: This framework applies
   an algorithmic approach EVIDENCE: A system uses al..." — el modelo
   alucina contenido no presente en el input.

4. **Errores sistematicos por tipo**: los casos SUPPORTS son fallados por
   4/4 workers (s-002, s-003). Los casos UNRELATED son los mas faciles
   (u-003: 0/4 wrong). El modelo no reconoce soporte directo pero si
   detecta irrelevancia topica — capacidad parcial, no suficiente.

5. **Ensemble no ayuda**: la diversidad de prompts no produce diversidad
   de errores porque el modelo falla por la misma razon estructural
   (falta de comprension semantica) en los mismos casos. Un ensemble
   no puede compensar una limitacion de capacidad fundamental del modelo.

6. **El modelo es 2B ternario**: BitNet-b1.58-2B-4T usa pesos ternarios
   (-1, 0, 1) en una arquitectura 2B. Esta cuantizacion extrema reduce
   la capacidad de razonamiento semantico. El modelo puede seguir
   patrones de formato pero no puede razonar sobre relaciones semanticas
   entre textos.

## Resolution

1. **ADR-0031 deprecado**: el ADR queda con estado "Deprecated" y
   referencia a este postmortem. La razon: el modelo seleccionado
   (BitNet-b1.58-2B-4T) no cumple el criterio de capacidad minima
   necesario para Fase 1.

2. **LLMSupport desacoplado del pipeline**: el wiring en `bootstrap.py`
   queda comentado con referencia a PM-003 y EXP-010. El codigo de
   LLMSupport, SemanticEnsemble, SemanticAssessmentAdapter se preserva
   como experimento documentado, no como componente activo.

3. **Scripts de pilot preservados**: `run_semantic_pilot.py`,
   `run_ensemble_pilot.py`, `run_llm_support_pilot.py` quedan con
   header explicando que son experimentos deprecados con referencia
   a PM-003 y EXP-010.

4. **No se eliminan los contratos**: `SemanticAssessment` y
   `SEMANTIC_RELATIONS` en `kernel/state.py` se preservan. Si un futuro
   modelo con mayor capacidad (7B+ o MoE) se evalua, los contratos
   ya existen y la frontera arquitectonica esta validada.

## Prevention

1. **No reintentar BitNet-b1.58-2B-4T para tareas semanticas**: este
   postmortem documenta con datos que el modelo no tiene capacidad
   suficiente. Reintentar sin un modelo diferente es desperdicio.

2. **Criterio de capacidad minima para Fase 2**: antes de reactivar
   LLMSupport, cualquier modelo nuevo debe demostrar >60% accuracy en
   el dataset de 12 pares claim-evidence (EXP-010). El dataset esta
   congelado en `scripts/run_semantic_pilot.py`.

3. **Modelo minimo recomendado**: RES-007 sugiere 7B como minimo para
   razonamiento semantico. BitNet-b1.58-2B-4T esta por debajo del
   umbral. Cualquier reevaluacion debe usar un modelo >=7B o un MoE
   con expertos especializados.

4. **Separar "sigue formato" de "comprende semantica"**: el pilot 2
   mostro 100% protocolo pero 33% accuracy. Estos son ortogonales.
   Un modelo puede seguir formato sin comprender semantica. Medir
   ambos por separado.

5. **No asumir que ensemble compensa capacidad**: el pilot 3 confirmo
   que un ensemble de modelos con la misma limitacion de capacidad
   no mejora. La diversidad de prompts no genera diversidad de errores
   cuando el problema es de capacidad, no de perspectiva.

## Confirmacion posterior (EXP-015, 2026-08-16)

Tras el descubrimiento de POST-001 (contaminacion de protocolo en
Llama3.2, Nemotron, Ministral, Qwen3.5), se re-evaluo BitNet en el
benchmark v2 completo (55 casos, 10 categorias diagnosticas) con
protocolo corregido (max_tokens=128, parser estricto, sin truncamiento).

Resultado: **BitNet NO fue contaminado por el protocolo**. Su
semantic_pilot original uso max_tokens=256 (sin truncamiento) y el
parser encontraba la relacion en el texto (no necesitaba default).

| Config | Historico (12 casos) | Controlado (55 casos) |
|--------|---------------------|----------------------|
| single | 33.3% | 29.1% |
| ensemble_2 | 33.3% | 36.4% |
| ensemble_4 | 41.7% | 30.9% |

La condena de PM-003 **se sostiene**. BitNet esta ~30 puntos por
debajo de Llama3.2 (el siguiente peor modelo) y ~50 puntos por
debajo de Qwen3. Ver EXP-015 para detalles completos.

## Confirmacion multidimensional exhaustiva (EXP-017, 2026-08-17)

Se realizo un barrido factorial completo aislando 3 dimensiones:

1. **Carga cognitiva**: zero-shot GBNF, few-shot GBNF, binary cascading
2. **Decoding**: temperature (0.0 vs 0.2), top_k (1 vs 20),
   repeat_penalty (1.0 vs 1.1)
3. **Ensemble**: single, ensemble_2, ensemble_4

**7 configuraciones exploradas, ninguna supero el 29.1%**:

| Configuracion | Accuracy |
|--------------|----------|
| Zero-Shot GBNF (temp=0.0) | 25.4% |
| Zero-Shot GBNF (temp=0.2, top_k=20) | 21.8% |
| **Few-Shot GBNF (temp=0.0, rep=1.0) [optimo]** | **29.1%** |
| Few-Shot GBNF (temp=0.2, top_k=20) | 25.4% |
| Binary Cascading / One-vs-All | 27.3% |
| Ensemble_2 (voting) | 27.3% |
| Ensemble_4 (voting) | 27.3% |

### Perfil cognitivo definitivo (mejor regimen, 55 casos)

| Categoria | BitNet | Qwen3.5 4B |
|-----------|--------|------------|
| direct_evidence | **0.0%** | 100.0% |
| paraphrase | **0.0%** | 100.0% |
| partial_support | 16.7% | 66.7% |
| explicit_contradiction | 80.0% | 100.0% |
| implicit_contradiction | 40.0% | 100.0% |
| negation | 50.0% | 100.0% |
| over_specificity | 20.0% | 80.0% |
| wrong_subject | 60.0% | 100.0% |
| wrong_context | 40.0% | 60.0% |
| adversarial | **0.0%** | 83.3% |
| **TOTAL** | **29.1%** | **89.1%** |

### Hallazgos tecnicos adicionales

1. **repeat_penalty=1.1 distorsiona BitNet**: penaliza SUPPORTS
   (logit mas alto que aparece en el prompt) y empuja a PARTIAL.
   repeat_penalty=1.0 es obligatorio para este modelo.

2. **Chain-of-Thought es contraproducente**: BitNet alucina
   negaciones ("is not required") que no existen en la evidencia.
   El CoT no ayuda porque el modelo copia fragmentos de los ejemplos.

3. **json_schema vs GBNF directo**: json_schema (JSON completo con
   worker, relation, confidence) satura la capacidad del modelo.
   GBNF directo a 1 token (solo la palabra de la relacion) es
   optimo: minima carga cognitiva.

4. **El ensemble confirma correlacion de errores**: los 4 workers
   votan igual en la mayoria de los casos. El voting no corrige
   errores porque comparten el mismo sesgo.

### Veredicto final

**La reduccion de precision a 1.58 bits destruye justamente la
capacidad de entailment que este componente necesita.** BitNet
tiene 0% en direct_evidence y paraphrase (las categorias que
requieren reconocer que dos textos con vocabulario diferente
pueden significar lo mismo).

Su unica fortaleza residual (80% en explicit_contradiction) es
redundante: Qwen3.5 alcanza 100% en esa misma categoria.

**PM-003 se sostiene de forma definitiva y multidimensional.**
No existe ningun regimen de inferencia que haga a BitNet viable
como evaluador semantico. Ver EXP-017 para detalles completos.

## Confirmacion con NLI reframing + logit ensemble (EXP-018, 2026-08-18)

EXP-017 documento que en 12/12 casos SUPPORTS, ninguno de los 5
regimenes emitio SUPPORTS. EXP-018 investigo si este sesgo era de
**token-etiqueta** (el token "SUPPORTS" es antinatural para el modelo)
o de **capacidad subyacente** (el modelo no reconoce entailment).

### Resultados de EXP-018

| Configuracion | Accuracy | vs EXP-017 |
|--------------|----------|------------|
| Techo single regime (EXP-017) | 29.1% (16/55) | — |
| NLI 4-way single (greedy) | 29.1% (16/55) | 0.0% |
| **Logit ensemble (fs0 + nli4 + bin, logsumexp)** | **38.2% (21/55)** | **+9.1%** |
| **Logit ensemble 4-regimen calibrado** | **40.0% (22/55)** | **+10.9%** |

### Hallazgos clave

1. **NLI reframing rompe la SUPPORTS wall**: reformulando a
   TRUE/FALSE/CANNOT_TELL, los 3 regimenes NLI aciertan 12/12 SUPPORTS.
   El sesgo era de token, no de capacidad subyacente. BitNet **si
   tiene** capacidad de entailment cuando se le pide que diga "TRUE"
   en lugar de "SUPPORTS".

2. **Logprobs revelan informacion oculta**: en casos de contradiccion
   explicita (ec-001, n-001), el token `False` tiene mayor logprob que
   `True`, pero el GBNF fuerza `TRUE` como greedy. El modelo distingue
   contradiccion de soporte en los logprobs, pero el grammar
   constraint selecciona el token equivocado.

3. **Logit ensemble captura complementariedad**: fs0 detecta
   contradicciones, NLI detecta soporte, bin detecta partial. El
   logsumexp preserva la confianza. El voting la destruia.

### Por que 50% no es alcanzable

El NLI reframing gana entailment pero **pierde relevancia detection**:

| Categoria | EXP-017 | EXP-018 | Delta |
|-----------|---------|---------|-------|
| direct_evidence | 0.0% | 100.0% | +100.0% |
| paraphrase | 0.0% | 83.3% | +83.3% |
| wrong_subject | 60.0% | 0.0% | -60.0% |
| wrong_context | 40.0% | 0.0% | -40.0% |
| partial_support | 16.7% | 0.0% | -16.7% |

El trade-off es fundamental: los regimenes NLI dicen YES/TRUE para
todo. No hay framing que recupere entailment y relevancia
simultaneamente en un modelo 2B ternario.

### Revision del veredicto de PM-003

La afirmacion original "la cuantizacion 1.58 bits destruye la capacidad
de entailment" debe **matizarse**:

- BitNet **si tiene** capacidad de entailment (NLI reframing lo demuestra)
- Lo que se observa debilitado es:
  1. La capacidad de usar etiquetas artificiales como tokens de salida
  2. La capacidad de distinguir relevancia de soporte (NLI dice YES para todo)
  3. La capacidad de emitir PARTIAL con confianza

La causa de estas debilidades no esta aislada: puede ser cuantizacion,
capacidad del modelo 2B, entrenamiento/instruction tuning, framing de
la tarea, representacion linguistica de UNRELATED/PARTIAL/CANNOT_TELL,
dificultad de distinguir truth evaluation de evidence relevance, o una
combinacion. Los experimentos demuestran el **que** (capacidades
debilitadas), no el **porque** (causa raiz).

**BitNet sigue sin ser viable** como evaluador semantico (40% < 60%
minimo), pero la razon es un trade-off entre entailment y relevancia,
no una destruccion total del entailment. Ver EXP-018 para detalles.

## Descomposicion relevance x entailment (EXP-019, 2026-08-18)

EXP-018 mostro que BitNet tiene entailment pero debilidad en relevance
detection. EXP-019 investigo si BitNet puede resolver ambas capacidades
por separado mediante un sistema de 2 etapas con decision layer
deterministico.

### Arquitectura

```
  STAGE 1: RELEVANCE (YES/NO)
       |
       +-- NO --> UNRELATED
       +-- YES --> STAGE 2: ENTAILMENT (TRUE/FALSE/PARTIALLY)
                        |
                        +-- TRUE --> SUPPORTS
                        +-- FALSE --> CONTRADICTS
                        +-- PARTIALLY --> PARTIAL
```

### Resultados clave

| Capacidad | Medicion | Estado |
|-----------|----------|--------|
| Entailment binario (TRUE/FALSE) | 12/12 SUPPORTS, 5/5 explicit_contradicts | Fuerte |
| Relevance detection (claro) | 100% wrong_subject + wrong_context (greedy) | Fuerte |
| Relevance detection (sutil) | ws-003, wc-003 pasan el gate | Debil |
| Granularity assessment (PARTIAL) | 0/17 en todos los experimentos | Ausente |

**BitNet detecta 100% de casos irrelevantes claros** cuando se le
pregunta directamente. La debilidad no es incapacidad total de
detectar relevancia — es threshold miscalibrado y casos sutiles.

### Techo alcanzado

| Configuracion | Accuracy |
|--------------|----------|
| EXP-017 techo single | 29.1% (16/55) |
| EXP-018 logit ensemble | 40.0% (22/55) |
| **EXP-019 hybrid (ensemble + relevance gate)** | **43.6% (24/55)** |

El hybrid combina el logit ensemble de EXP-018 con un relevance gate
que override SUPPORTS -> UNRELATED cuando el relevance logprob diff
<= -1.8. Recupera wrong_subject (0% -> 80%) pero pierde 2 paraphrase.

### Frontera de BitNet

La frontera esta en **granularity assessment** (PARTIAL) y **relevance
sutil** (mismo topico, diferente entidad), no en entailment ni en
relevance detection claro. Los 4 aciertos que faltan para 50% estan
todos en PARTIAL (partial_support 0/6, over_specificity 0/5,
adversarial 1/6).

Esto es consistente con la caracterizacion:

> "modelo capaz de evaluar relacion proposicional, pero incapaz de
> modelar adecuadamente las condiciones de aplicabilidad de esa
> relacion."

Las "condiciones de aplicabilidad" tienen dos componentes:
1. Relevance (es esta evidencia sobre el sujeto/contexto correcto?) —
   moderado, falla en casos sutiles
2. Granularity (soporta el evidence todo el claim o solo parte?) —
   ausente

**PM-003 se sostiene** (43.6% < 60% minimo). La frontera esta mas
especifica de lo que se pensaba: no es "entailment destruido" ni
"relevance destruido", es granularity assessment ausente + relevance
sutil debil. Ver EXP-019 para detalles.

## Granularity probe y verificacion de ausencia (EXP-020, 2026-08-18)

EXP-020 aislo granularity assessment con casos minimos controlados
(22 casos, sin vocabulario complejo, sin relevance confounders) y
probo atomic decomposition (descomponer claims en proposiciones
atomicas, evaluar cada una, agregar deterministamente).

### Resultados clave

| Fase | Regimen | Accuracy |
|------|---------|----------|
| Fase 1: directo | NLI 4-way | 27.3% (6/22) |
| Fase 1: directo | NLI Cascading | 54.5% (12/22) — sesgo de token |
| Fase 1: directo | Direct FULL/PARTIAL | 9.1% (2/22) |
| Fase 2: atomic | Aggregator (decomposer arreglado) | 33.3% (5/15) |
| Fase 2: atomic | **Atomic FALSE accuracy** | **0%** |

### El hallazgo critico

**BitNet hace keyword matching parcial, no verificacion composicional.**

Cuando el atom es "The system supports encryption in transit" y el
evidence es "The system supports encryption at rest", BitNet dice TRUE
porque ve "The system supports encryption" (match parcial) y no
verifica que "in transit" vs "at rest" son diferentes.

Atomic FALSE accuracy: **0%**. El modelo no puede decir FALSE cuando
una proposicion no esta confirmada pero comparte keywords con el
evidence.

### Frontera definitiva (bajo el protocolo actual)

La cadena EXP-017 -> 018 -> 019 -> 020 aisla progresivamente la
frontera:

| Capacidad | EXP | Estado |
|-----------|-----|--------|
| Entailment binario (TRUE/FALSE) | 018 | Fuerte |
| Relevance detection (claro) | 019 | Fuerte |
| Relevance detection (sutil) | 019 | Debil |
| **Verificacion de ausencia** | **020** | **Ausente** |
| Granularity assessment (PARTIAL) | 017-020 | Ausente |

**La frontera es la verificacion de ausencia.** Sin ella, BitNet no
puede emitir PARTIAL, UNRELATED (en casos sutiles), ni CONTRADICTS
implicita. La atomic decomposition no funciona porque el aggregator
no puede componer señales correctas partiendo de señales incorrectas
(todas TRUE).

### Conclusion

**PM-003 se sostiene definitivamente (bajo el protocolo actual).**
BitNet no es viable como evaluador semantico generalista. La causa
no esta aislada (puede ser cuantizacion, capacidad 2B, entrenamiento,
framing, o combinacion).

BitNet puede conservarse como **extractor barato de señales semanticas
elementales**:
- Relevance detection claro: 100% (EXP-019)
- Entailment binario: 12/12 SUPPORTS (EXP-018)

La autoridad queda en el sistema (contratos y politicas deterministas).
Ver EXP-020 para detalles.

## Falsation probe: verificacion de ausencia (EXP-021, 2026-08-18)

EXP-021 es el experimento de falsacion final. Tres condiciones minimas,
una sola pregunta: puede BitNet decir FALSE cuando debe?

### Resultado

| Condicion | Expected | Greedy | Accuracy |
|-----------|----------|--------|----------|
| implicit_absence (A+B, claim A+B+C) | FALSE | TRUE (6/6) | **0%** |
| explicit_negation (A+B, claim A+B+NOT-C) | TRUE | TRUE (6/6) | 100% |
| total_absence (A+B, claim C) | FALSE | TRUE (6/6) | **0%** |

**FALSE accuracy: 0/12 (0.0%).** BitNet dice TRUE para todo lo
no-soportado. Margen logP(TRUE)-logP(FALSE) siempre positivo (medio
+0.776, min +0.365). El modelo esta seguro de sus respuestas
incorrectas.

### Veredicto final

**H0 rechazada. H1 confirmada.** BitNet no puede usar ausencia de
evidencia como condicion negativa para una inferencia composicional.
Incapacidad operacional confirmada.

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

### Conclusion de la cadena experimental (EXP-017 -> 021)

La pregunta deja de ser: "Como hacemos que BitNet sea un buen juez?"

Y pasa a ser: "Donde es economicamente optimo utilizar la senal que
BitNet si puede producir?"

BitNet como **extractor barato de senales semanticas elementales**
(relevance claro, entailment binario), con la autoridad en el sistema.
No como evaluador semantico completo.

### Formulacion final de PM-003

**PM-003 — BitNet como Semantic Assessor: REJECTED (definitivo)**

Los experimentos EXP-017 a EXP-024 no encontraron señal semantica
explotable en BitNet 1.58b, ni en el greedy decoding, ni en la
distribucion de probabilidades, ni bajo ningun grammar.

**EXP-024 (definitivo):** P(TRUE|SUPPORTS) = 0.4000, P(TRUE|CONTRADICTS)
= 0.3983, delta = +0.0017. La distribucion de probabilidades no cambia
sistematicamente segun la relacion semantica. BitNet no discrimina
semanticamente. No hay señal oculta en los logprobs.

**Caveat metodologico (EXP-023):** La gramatica de decodificacion
(GBNF) es una variable experimental de primer orden para BitNet.
EXP-018 y EXP-022 usaron grammars estrictos (sin espacios), produciendo
TRUE para todo. EXP-019, 020, 021 usaron grammars permisivos (con
espacios), produciendo FALSE para todo. Las comparaciones entre
experimentos con diferentes grammars estan confundidas. EXP-024
controlo esta variable y confirmo que el grammar no afecta la
distribucion subyacente (siempre ~40% TRUE, ~60% FALSE).

**Decision:** BitNet no tiene caso de uso en el pipeline semantico.
No como juez generalista, no como worker especializado, no como
extractor de señales elementales. La distribucion de probabilidades
no contiene señal semantica explotable.

**Excepcion unica:** relevance detection clara (wrong_subject con
entidades completamente diferentes, EXP-019: 100%). Pero EXP-024
muestra que esto no se refleja en P(TRUE)/P(FALSE) — probablemente
funciona por ausencia total de overlap de keywords, no por
discriminacion semantica. No generaliza a distinciones finas.

### Consecuencia arquitectonica

La cadena experimental reforzo la separacion fundamental:

```
             LLMs
              |
              v
       semantic signals
              |
              v
      deterministic layer
              |
        +-----+-----+
        | Contracts  |
        | Policies   |
        | Invariants |
        +-----+-----+
              |
              v
          AUTHORITY
```

BitNet puede generar: "hay evidencia relacionada" (senal).
No deberia decidir: "Evidence Contract satisfied = TRUE" (autoridad).

La arquitectura no necesita que un LLM sea un juez omnisciente.
Necesita modelos capaces de producir senales especificas que el
sistema pueda interpretar y gobernar mediante contratos.

La cruzada BitNet termino demostrando algo mas interesante que si
BitNet llega o no al 50%: **la arquitectura no necesita un juez
omnisciente, necesita senales gobernables.**

**PM-003 cerrado.**
