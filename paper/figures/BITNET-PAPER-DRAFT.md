# BitNet b1.58-2B-4T como Evaluador Semántico: Una Investigación Experimental Controlada

## Abstract

Este documento reporta una investigación experimental exhaustiva sobre la viabilidad de utilizar BitNet b1.58-2B-4T — un modelo de lenguaje cuantizado a 1.58 bits (ternario) — como componente de evaluación semántica dentro de un pipeline de Retrieval-Augmented Generation (RAG). A través de ocho experimentos progresivamente controlados (EXP-017 a EXP-024), aislamos las variables que confundían las mediciones iniciales y llegamos a una conclusión definitiva: **la distribución de probabilidades de BitNet no contiene señal semántica explotable**, ni en el greedy decoding, ni en los logprobs, ni bajo ninguna configuración de grammar de decodificación. El comportamiento observable del modelo es consistente con matching holístico por overlap lexical, no con discriminación semántica. Reportamos adicionalmente el hallazgo metodológico de que la gramática de decodificación (GBNF) es una variable experimental de primer orden para este modelo, capaz de invertir completamente la distribución de outputs.

---

## 1. Contexto y Finalidad

### 1.1 Entorno experimental

La investigación se condujo en un pipeline de Agentic RAG diseñado para evaluar afirmaciones (claims) contra evidencia recuperada. El componente central de este pipeline es el **Semantic Assessment**: determinar si una pieza de evidencia soporta, contradice, parcialmente soporta, o no está relacionada con un claim.

Las cuatro relaciones semánticas canónicas son:

```
SUPPORTS     — la evidencia demuestra el claim
PARTIAL      — la evidencia soporta parte del claim pero no todo
CONTRADICTS  — la evidencia es incompatible con el claim
UNRELATED    — la evidencia no trata sobre el mismo tema/contexto
```

### 1.2 Motivación arquitectónica

El pipeline requiere múltiples evaluaciones semánticas por consulta. Usar un modelo grande (7B+) para cada evaluación es costoso. La hipótesis inicial era que BitNet b1.58-2B-4T, debido a su tamaño reducido (2B parámetros) y cuantización extrema (1.58 bits/parámetro), podría servir como **extractor barato de señales semánticas elementales** dentro de una arquitectura Mixture-of-Experts (MoE):

```
                Semantic Assessment
                       │
             ┌─────────┼─────────┐
             │         │         │
        Relevance  Entailment  Granularity
             │         │         │
          BitNet    BitNet?    Qwen3.5
             │         │         │
             └─────────┼─────────┘
                       │
              deterministic policy
```

BitNet entraría donde demostráramos experimentalmente que aporta una señal incremental estable. No por ser barato, no por ser rápido, no porque una configuración obtuviera 12/12 — sino porque la distribución de probabilidades contuviera información semántica explotable.

### 1.3 Infraestructura

- **Modelo:** BitNet-b1.58-2B-4T (2B parámetros, pesos ternarios {-1, 0, +1})
- **Backend:** llama-server (bitnet.cpp), API compatible con OpenAI en `/completion`
- **Hardware:** CPU (4 threads), sin GPU
- **Decodificación:** GBNF grammars para restringir outputs, logprobs via `n_probs`
- **Benchmark:** 55 casos balanceados en 10 categorías (direct_evidence, paraphrase, partial_support, explicit_contradiction, implicit_contradiction, negation, over_specificity, wrong_subject, wrong_context, adversarial)

---

## 2. Cadena Experimental

### 2.1 Visión general

```
EXP-017  →  EXP-018  →  EXP-019  →  EXP-020  →  EXP-021  →  EXP-022  →  EXP-023  →  EXP-024
  │           │           │           │           │           │           │           │
  ▼           ▼           ▼           ▼           ▼           ▼           ▼           ▼
baseline   NLI         relevance   atomic      ausencia    micro-      grammar     semantic
pobre      reframing    gate        decompos.   0/12 FALSE  coliseum    sensitivity  discrimination
29.1%      40.0%       43.6%       33.3%       →frontera   29.1%       TRUE↔FALSE   delta=0.0017
                       peak        →keyword     →ausencia               inversion    →NO señal
                                   matching     →PM-003
```

Cada experimento respondió una pregunta y abrió la siguiente.

### 2.2 EXP-017: Baseline multidimensional

**Pregunta:** ¿Qué puede hacer BitNet con etiquetas artificiales (SUPPORTS/CONTRADICTS/PARTIAL/UNRELATED)?

**Resultado:** 29.1% accuracy. SUPPORTS wall: 0/12 SUPPORTS correctos. El modelo nunca emite SUPPORTS como token.

**Interpretación (corregida):** El modelo tiene sesgo contra tokens artificiales multi-palabra. Esto motivó NLI reframing.

### 2.3 EXP-018: NLI Reframing + Logit Ensemble

**Pregunta:** ¿Si reformulamos a tokens naturales (TRUE/FALSE/CANNOT_TELL), se rompe la SUPPORTS wall?

**Resultado:** 40.0% con logit ensemble. 12/12 SUPPORTS en greedy.

**Interpretación inicial:** "El sesgo era de token, no de capacidad. BitNet tiene entailment."

**Interpretación corregida (EXP-023):** Los 12/12 SUPPORTS son consistentes con TRUE universal inducido por grammar estricto. El grammar forzó el token `TRUE` en 55/55 casos independientemente de la semántica. **No hay evidencia de discriminación semántica.**

### 2.4 EXP-019: Relevance Gate + Hybrid Ensemble

**Pregunta:** ¿Podemos separar relevance detection de entailment y combinarlos?

**Resultado:** 43.6% (peak histórico). Relevance gate: 100% en wrong_subject, 80% en wrong_context.

**Interpretación inicial:** "BitNet tiene relevance detection clara."

**Interpretación corregida (EXP-024):** La relevance detection funciona por **separabilidad lexical extrema**, no por discriminación semántica. Cuando dos textos no comparten vocabulario relevante (Product A vs Product B), BitNet responde diferencialmente. Esto no generaliza a distinciones semánticas finas (NIST vs ISO 27001).

### 2.5 EXP-020: Granularity Probe + Atomic Decomposition

**Pregunta:** ¿Podemos descomponer claims en proposiciones atómicas y verificar composicionalmente?

**Resultado:** 33.3% con atomic decomposition. Atomic FALSE accuracy: 0%.

**Interpretación:** El comportamiento observable es consistente con keyword matching holístico. BitNet dice TRUE para cualquier proposición que comparta keywords con el evidence, sin verificar ausencia.

### 2.6 EXP-021: Absence Detection Falsation Probe

**Pregunta:** ¿Puede BitNet emitir FALSE cuando la información requerida está ausente?

**Resultado:** 0/12 FALSE correctos. TRUE en 12/12 casos. Margen logprob siempre positivo (+0.776 medio).

**Interpretación inicial:** "BitNet no puede usar ausencia como condición negativa."

**Interpretación corregida (EXP-023):** Este experimento usó grammar estricto, que fuerza TRUE. BitNet sí emite FALSE naturalmente bajo grammar permisivo (53/55 en EXP-023). El hallazgo correcto es: **el token FALSE no aparece como respuesta a ausencia bajo este framing particular**, no que BitNet no pueda emitir FALSE.

### 2.7 EXP-022: Microcoliseum Especializado

**Pregunta:** ¿Un ensemble de 4 workers especializados + judge determinístico supera al mejor régimen individual?

**Resultado:** 29.1%. Judge determinístico agrega +5.5% sobre ensemble. Contradicciones: 20%→60%.

**Interpretación:** La especialización por worker funciona parcialmente, pero el trade-off del judge (boost a CONTRADICTS daña PARTIAL) confirma que las señales individuales no son confiables.

### 2.8 EXP-023: Grammar Sensitivity Probe

**Pregunta:** ¿El grammar GBNF afecta el comportamiento de BitNet?

**Resultado:** **Inversión completa.**

| Condición | Token dominante | Distribución |
|-----------|-----------------|--------------|
| G1 estricto | `TRUE` x55 | 55/55 SUPPORTS |
| G2 permisivo | `False` x40, `FALSE` x13 | 53/55 CONTRADICTS |
| G3 sin grammar | `False` x39, `FALSE` x13 | 53/55 CONTRADICTS |

G2 = G3 (McNemar χ²=0.00). El grammar permisivo no restringe nada que el modelo no fuera a hacer naturalmente. G1 fuerza un comportamiento artificial.

**Mecanismo:** Los tokens con y sin espacio (`"TRUE"` vs `" FALSE"`) son tokens diferentes con probabilidades diferentes. El grammar estricto fuerza el token sin espacio, cambiando la preferencia.

**Impacto retrospectivo:** Las comparaciones entre EXP-018 (estricto) y EXP-019/020/021 (permisivo) están confundidas por el grammar.

### 2.9 EXP-024: Semantic Discrimination × Decoding

**Pregunta:** ¿Existe señal semántica en la distribución de probabilidades, antes de imponer cualquier interfaz de decisión?

**Diseño:** 20 pares mínimos (mismo evidence, claims que difieren en un elemento semántico) × 3 grammars = 180 LLM calls. Seed fijo, temperature 0.0, n_probs=15.

**Resultado:**

| Expected | P(TRUE) | P(FALSE) | P(CANNOT) |
|----------|---------|----------|-----------|
| SUPPORTS | 0.4000 | 0.6000 | 0.0000 |
| CONTRADICTS | 0.3983 | 0.6017 | 0.0000 |
| PARTIAL | 0.4668 | 0.5332 | 0.0000 |
| UNRELATED | 0.3941 | 0.6059 | 0.0000 |

**Delta P(TRUE|SUPPORTS) - P(TRUE|CONTRADICTS) = +0.0017**

La probabilidad de TRUE es idéntica independientemente de la relación semántica. 0.0017 es ruido puro.

**Independiente del grammar:** los tres grammars producen distribuciones casi idénticas (~40% TRUE, ~60% FALSE).

**P(CANNOT_TELL) = 0.0000** en todos los casos: el modelo no tiene representación de incertidumbre.

**Sensibilidad a modificación semántica mínima:** solo 1-2 de 19 pares muestran diferencia > 0.05. Eso es azar.

---

## 3. Resultado Final: Mapa de Comportamiento

### 3.1 Mapa de capacidades

```
                    BitNet 1.58B
                         │
              ┌──────────┴──────────┐
              │                     │
       comportamiento         capacidades
        observable             semánticas
              │                     │
      ┌───────┼────────┐            │
      │       │        │            │
    TRUE    FALSE    grammar      ¿señal?
      │       │        │            │
      │       │        │           NO
      │       │        │            │
      └───────┴────────┴────────────┘
                         │
                  lexical behavior
                         │
              ┌──────────┴──────────┐
              │                     │
       alto overlap             bajo overlap
              │                     │
        respuestas             separación
        sesgadas               aparente
```

### 3.2 Lexical separability vs semantic relevance

El 100% de relevance detection en casos claros (EXP-019: wrong_subject) sigue siendo un resultado medido. Pero la interpretación correcta es:

**BitNet puede responder diferencialmente cuando existe una separación lexical extremadamente fuerte entre los dos textos.**

Por ejemplo:
- Product A vs Product B
- Framework X vs Framework Y

Si la evidencia comparte poco o ningún vocabulario relevante con el claim, BitNet produce una respuesta diferente. Esto **no constituye evidencia de semantic relevance general**. Es compatible con:

```
overlap alto → una respuesta (TRUE o FALSE, según grammar)
overlap bajo  → otra respuesta
```

Por lo tanto, clasificamos esto como **lexical separability**, no como **semantic relevance detection**. La distinción es importante:

| Concepto | Mecanismo | Generaliza |
|----------|-----------|------------|
| Semantic relevance detection | El modelo comprende que dos textos tratan sobre el mismo tema | Sí, a distinciones finas |
| Lexical separability | El modelo detecta ausencia/presencia de overlap de vocabulario | No, solo a separaciones extremas |

EXP-024 confirma esto: en los pares mínimos de subject (G4), donde el overlap es alto pero la entidad difiere (NIST vs ISO 27001), la distribución P(TRUE)/P(FALSE) no cambia. Solo funciona cuando el overlap es cercano a cero.

### 3.3 Implicancia arquitectónica

```
BitNet
   │
   X Semantic Assessment
   X Judge
   X Worker
   X Evidence classifier
   X MoE semantic expert
   │
   └── descartado
```

BitNet no tiene caso de uso en el pipeline semántico. No como juez generalista, no como worker especializado, no como extractor de señales elementales. La distribución de probabilidades no contiene señal semántica explotable.

---

## 4. Hallazgo Metodológico: Grammar como Variable de Primer Orden

### 4.1 El descubrimiento

Durante EXP-022 observamos que cambiar de grammar permisivo a estricto invertía completamente el comportamiento de BitNet (de FALSE x53 a TRUE x55). EXP-023 aisló esta variable:

- **Grammar estricto** (`root ::= "TRUE" | "FALSE" | "CANNOT_TELL"`): fuerza tokens sin espacio, BitNet prefiere `TRUE`
- **Grammar permisivo** (con `" TRUE"`, `" True"`, etc.): permite tokens con espacio, BitNet prefiere ` FALSE`
- **Sin grammar**: idéntico a permisivo

### 4.2 El mecanismo

Los tokens con y sin espacio son tokens diferentes en el vocabulario de BitNet:

- `"TRUE"` (sin espacio) → token ID X, con cierta probabilidad
- `" FALSE"` (con espacio) → token ID Y, con mayor probabilidad

El grammar estricto elimina el token Y del espacio de decodificación, forzando el token X. Esto cambia la preferencia del modelo de FALSE a TRUE.

### 4.3 Impacto en la reproducibilidad

Los experimentos de esta investigación usaron diferentes grammars:

| Experimento | Grammar | Comportamiento resultante |
|-------------|---------|---------------------------|
| EXP-017 | Labels artificiales | SUPPORTS wall |
| EXP-018 | Estricto | TRUE universal → "entailment" |
| EXP-019 | Permisivo | FALSE universal → "contradiction detection" |
| EXP-020 | Permisivo | FALSE universal |
| EXP-021 | Estricto | TRUE universal → "ausencia no detectada" |
| EXP-022 | Estricto | TRUE universal → "SUPPORTS" |
| EXP-023 | Controlado | Inversión demostrada |
| EXP-024 | Controlado | Independiente del grammar |

**Las comparaciones directas entre experimentos con diferentes grammars están confundidas.** EXP-024 controló esta variable y demostró que el grammar no afecta la distribución subyacente — solo afecta qué token se selecciona del argmax.

### 4.4 Recomendación

> La gramática de decodificación (GBNF) debe reportarse como variable experimental de primer orden para modelos cuantizados a 1.58 bits, equivalente a temperature o seed. Los grammars estrictos pueden inducir comportamientos que no reflejan la distribución natural del modelo.

---

## 5. Discusión

### 5.1 ¿Por qué no hay señal semántica?

No podemos atribuir la ausencia de señal a una causa única. Los candidatos son:

1. **Cuantización a 1.58 bits:** los pesos ternarios {-1, 0, +1} pueden destruir la resolución fina necesaria para discriminación semántica
2. **Tamaño del modelo (2B):** capacidad insuficiente para representar relaciones semánticas
3. **Entrenamiento:** el modelo puede no haber aprendido representaciones semánticas suficientes
4. **Combinación de los anteriores**

La investigación demuestra el **qué** (no hay señal semántica explotable), no el **por qué** (causa raíz).

### 5.2 ¿Por qué la distribución es ~40% TRUE / ~60% FALSE?

Bajo todos los grammars y todas las relaciones semánticas, la distribución se mantiene aproximadamente en 40% TRUE / 60% FALSE. Esto sugiere un sesgo de decodificación dominante que es independiente de la semántica del input. El modelo tiene una preferencia base por FALSE (o `" False"` con espacio), y el grammar estricto puede invertir esto forzando el token sin espacio.

### 5.3 ¿Por qué P(CANNOT_TELL) = 0.0000?

El modelo nunca asigna probabilidad a CANNOT_TELL, ni siquiera en casos UNRELATED donde la respuesta correcta sería "no puedo determinar". Esto sugiere que BitNet no tiene representación de incertidumbre epistémica: trata every input como si pudiera determinarlo.

### 5.4 La excepción: lexical separability

La única capacidad medible es lexical separability: cuando dos textos no comparten vocabulario relevante, BitNet produce respuestas diferencialles. Esto es útil para filtrar casos de wrong_subject extremo (Product A vs Product B), pero:

1. No generaliza a distinciones semánticas finas (NIST vs ISO 27001)
2. No se refleja en la distribución P(TRUE)/P(FALSE) (EXP-024)
3. Es consistente con un mecanismo de overlap de keywords, no con comprensión semántica

### 5.5 Limitaciones del estudio

1. **Un solo modelo:** solo evaluamos BitNet b1.58-2B-4T. No podemos generalizar a otros modelos cuantizados a 1.58 bits.
2. **Un solo benchmark:** semantic_assessment_v2.json (55 casos) + semantic_discrimination_v1.json (60 casos). Dominio: ciberseguridad/NIST.
3. **Un solo prompt template:** NLI 3a (TRUE/FALSE/CANNOT_TELL). Otros framings podrían producir resultados diferentes, aunque EXP-018 ya probó múltiples regímenes.
4. **Seed fija (42):** no medimos varianza entre seeds.

---

## 6. Conclusiones

### 6.1 Conclusión principal

**BitNet b1.58-2B-4T no contiene señal semántica explotable en su distribución de probabilidades.** La probabilidad de generar TRUE o FALSE es independiente de la relación semántica entre claim y evidence (delta = +0.0017). El comportamiento observable del modelo es consistente con matching holístico por overlap lexical, no con discriminación semántica.

### 6.2 Conclusión metodológica

**La gramática de decodificación (GBNF) es una variable experimental de primer orden para BitNet.** Grammars estrictos y permisivos producen comportamientos opuestos (TRUE universal vs FALSE universal). Los grammars estrictos inducen trayectorias de decodificación artificiales que no reflejan la distribución natural del modelo. Esta variable debe reportarse en todo experimento con modelos cuantizados a 1.58 bits.

### 6.3 Conclusión arquitectónica

**BitNet no tiene caso de uso en el pipeline semántico.** La arquitectura RAG no necesita un modelo barato para señales semánticas elementales si esas señales no contienen información semántica. La separación señal → interpretación → autoridad sigue siendo válida, pero BitNet no puede ocupar el rol de productor de señales semánticas.

### 6.4 Lo que la investigación sí encontró

A pesar del resultado negativo, la investigación produjo varios hallazgos de valor:

1. **Un protocolo experimental para aislar capacidad semántica de sesgo de decodificación** (EXP-023 + EXP-024)
2. **Evidencia de que el grammar GBNF puede invertir el comportamiento de un modelo** (EXP-023)
3. **Una distinción entre lexical separability y semantic relevance detection** (EXP-019 + EXP-024)
4. **Un benchmark de pares mínimos para detectar señal semántica** (semantic_discrimination_v1.json)
5. **Evidencia de que la ausencia de señal no es un problema de extracción sino de representación** (EXP-024)

### 6.5 La pregunta que cambió

Hasta EXP-022 estábamos intentando descubrir:
> "¿Qué arquitectura permite hacer que BitNet sea un buen semantic assessor?"

EXP-023 nos obligó a preguntar antes:
> "¿Qué comportamiento semántico está realmente presente en la distribución de BitNet antes de imponer una interfaz de decisión?"

EXP-024 respondió:
> "Ninguno. La distribución es plana respecto a la semántica."

Esa es una respuesta más útil que cualquier optimización de accuracy. Nos dice que no hay nada que optimizar.

---

## 7. Tabla de Experimentos

| EXP | Pregunta | Resultado | Interpretación final |
|-----|----------|-----------|---------------------|
| EXP-017 | ¿Baseline con labels artificiales? | 29.1%, SUPPORTS wall | Sesgo contra tokens multi-palabra |
| EXP-018 | ¿NLI reframing rompe la wall? | 40.0%, 12/12 SUPPORTS | Artefacto del grammar estricto (EXP-023) |
| EXP-019 | ¿Relevance gate + ensemble? | 43.6% (peak) | Lexical separability, no semantic relevance |
| EXP-020 | ¿Atomic decomposition? | 33.3%, atomic FALSE 0% | Keyword matching holístico |
| EXP-021 | ¿Detección de ausencia? | 0/12 FALSE | Confundido por grammar estricto |
| EXP-022 | ¿Microcoliseum especializado? | 29.1%, +5.5% judge | Especialización parcialmente artefacto |
| EXP-023 | ¿Grammar afecta comportamiento? | Inversión completa TRUE↔FALSE | Grammar es variable de primer orden |
| EXP-024 | ¿Hay señal semántica en logprobs? | Delta = +0.0017 | **No. No hay señal semántica.** |

---

## 8. Artefactos

| Archivo | Descripción |
|---------|-------------|
| `benchmarks/semantic_assessment_v2.json` | Benchmark principal (55 casos, 10 categorías) |
| `benchmarks/semantic_discrimination_v1.json` | Pares mínimos (20 pares × 3 variantes = 60 casos) |
| `runners/run_bitnet_nli_reframing.py` | EXP-018 runner |
| `runners/run_bitnet_logit_ensemble.py` | EXP-018 logit ensemble |
| `runners/run_bitnet_relevance_entailment_decomposition.py` | EXP-019 runner |
| `runners/run_bitnet_granularity_probe.py` | EXP-020 runner |
| `runners/run_bitnet_absence_detection.py` | EXP-021 runner |
| `runners/run_bitnet_microcoliseum.py` | EXP-022 runner |
| `runners/run_bitnet_grammar_sensitivity.py` | EXP-023 runner |
| `runners/run_bitnet_semantic_discrimination.py` | EXP-024 runner |
| `experiments/EXP-001-bitnet/PM-003-bitnet-semantic-capacity.md` | Documento maestro de la investigación |

---

## 9. Referencias internas

- PM-003: BitNet Semantic Capacity (closed: REJECTED)
- EXP-017 a EXP-024: experimentos individuales con documentación completa
- POST-001: Protocol Contamination (análisis de contaminación de prompts)

---

## Apéndice A: Configuración experimental estándar

```
Modelo:          BitNet-b1.58-2B-4T
Backend:         llama-server (bitnet.cpp)
Endpoint:        http://127.0.0.1:{port}/completion
Hardware:        CPU, 4 threads, sin GPU
Temperature:     0.0 (greedy)
Seed:            42 (EXP-023, EXP-024)
num_predict:     4-6 tokens
n_probs:         10-15 (top-N logprobs)
Grammar:         GBNF (variable experimental)
Prompt:          NLI 3a few-shot (TRUE/FALSE/CANNOT_TELL)
```

## Apéndice B: Glosario

| Término | Definición |
|---------|------------|
| **Grammar estricto** | GBNF que solo permite la forma canónica del token (sin espacios ni variantes de caso) |
| **Grammar permisivo** | GBNF que permite espacios y variantes de caso (`" TRUE"`, `" True"`, `" true"`) |
| **Lexical separability** | Capacidad de responder diferencialmente cuando dos textos no comparten vocabulario relevante |
| **Semantic relevance detection** | Capacidad de determinar si dos textos tratan sobre el mismo tema, independientemente del overlap lexical |
| **Logit ensemble** | Agregación de logprobs de múltiples regímenes via logsumexp |
| **Pares mínimos** | Casos que comparten evidence pero difieren en un elemento semántico del claim |
| **SUPPORTS wall** | Sesgo observado en EXP-017 donde BitNet nunca emitía SUPPORTS como token |
