---
id: RES-004
category: research
status: accepted
created: 2026-07-28
updated: 2026-07-28
author: human
components: [llm_support, trace_sink, model_provider, policy_engine, execution_state]
tags: [architecture, query-time, llm-support, observer, hypotheses, reactive, incremental, passive-advisory, small-model, cpu-parallel]
related: [RES-001, RES-002, RES-003, RES-016, ADR-0005, ADR-0006, ADR-0007, ADR-0013, ADR-0014, ADR-0020, ADR-0031, BM-004]
supersedes: null
superseded_by: null
---

# RES-004 - LLMSupport: observador paralelo de hipotesis

> Extracto de RES-003 (seccionado 2026-07-28). RES-003 contenia dos preocupaciones con
> lifecycles y dependencias independientes:
> - **RES-003** — Knowledge Consumer / consumo de Warm Artifacts (depende de RES-001 y RES-002)
> - **RES-004** — LLMSupport / observador paralelo (este documento; depende solo de TraceSink y ModelProvider)

## Topic

Un componente LLM transversal que corre paralelo al pipeline, observa eventos del runtime de forma reactiva e incremental, y produce hipotesis sin bloquear ni decidir.

## Sources

- RES-003 (origen del contenido): seccion 3 LLMSupport, comparativas y migracion asociada
- ADR-0005: Observability como substrato transversal
- ADR-0007: ModelProvider
- ADR-0013: Policy Engine
- ADR-0014: Inyeccion de dependencias
- ADR-0020: Ownership de decisiones y contrato de ejecucion observable (P16, P17)
- BM-004: estado del Consumer al momento del research

---

## 1. Idea central

Un componente LLM que corre **paralelo al pipeline**, tomando la temperatura del sistema. Mientras el pipeline ejecuta su cadena secuencial:

```
Query
   |
Retrieval
   |
Reranker
   |
VERIFY
   |
LLM (generation)
```

El LLMSupport observa simultaneamente:

```
                LLMSupport
                    |
                    v
Query -------------->|
                    |
Retrieval events --->|
                    |
Reranker events ---->|
                    |
VERIFY events ------>|
                    |
LLM generation ----->|
```

## 2. Principios invariantes

El LLMSupport:

- **Nunca bloquea** — corre en paralelo (async/thread). El pipeline nunca espera al LLMSupport.
- **Nunca decide** — no invoca capabilities, no modifica `ExecutionState`, no sobrescribe decisiones.
- **Nunca reemplaza** — no sustituye ASSESS, VERIFY, ni al Policy Engine.
- **Simplemente observa** el estado del sistema y produce hipotesis.

Estos principios alinean directamente con:

| Principio | Como aplica |
|---|---|
| **P17** (ADR-0020) | La observabilidad no cambia el comportamiento — LLMSupport empieza 100% pasivo |
| **P16** (ADR-0020) | Ownership de decisiones — LLMSupport produce opiniones, el Policy Engine decide |
| **P14** | Una responsabilidad por eslabon — LLMSupport observa, no ejecuta |
| **P9** | Determinismo en el control, razonamiento en el lenguaje — LLMSupport razona, no controla |
| **P3** | Observabilidad antes que magia — toda hipotesis es trazable |
| **P4** | Medible antes que inteligente — pasivo primero, medir, luego habilitar influencia |

## 3. Que produce

No produce decisiones. Produce **hipotesis**.

Ejemplo:

```
Pipeline:
  BM25: Documento A
  Embedding: Documento B
  Reranker: Confianza 0.42

Mientras tanto LLMSupport razona:

  "La evidencia es muy pobre."
  "No encontre consenso entre BM25 y embeddings."
  "La consulta parece referirse a un documento especifico."
  "La respuesta probablemente necesite otro retrieval."
```

Esas son hipotesis. No ordenes.

## 4. Contrato de hipotesis

El LLMSupport produce un nuevo tipo de output: `Hypothesis`.

```json
{
  "suggestion": "RETRY_RETRIEVAL",
  "confidence": 0.71,
  "reasoning": "BM25 y embeddings no coinciden; reranker confidence 0.42 indica evidencia debil",
  "stage": "post_reranker",
  "run_id": "abc123"
}
```

- No es `EvaluationSignal` (no gatea, no produce pass/fail).
- No es `ActionDecision` (no ejecuta, no tiene `capability_ref`).
- Es una **opinion estructurada** con confidence y razonamiento.

## 5. Modelo reactivo incremental

El LLMSupport funciona incrementalmente, reactivo a eventos:

```
Evento 1: Query recibida
  -> piensa: analiza la pregunta
  -> hipotesis inicial

Evento 2: Llegaron candidatos (retrieval)
  -> actualiza hipotesis

Evento 3: Llego reranker
  -> actualiza hipotesis

Evento 4: LLM empezo a generar
  -> actualiza hipotesis
```

Es un sistema reactivo basado en eventos. Cada evento del pipeline dispara una actualizacion de la hipotesis. No necesita ver todo el estado para producir una opinion util.

## 6. No agrega latencia

Como corre en paralelo, mientras el reranker esta trabajando, el LLMSupport ya puede haber terminado.

```
t=0 ms    Query llega
          -> LLMSupport empieza a analizar la pregunta
          -> Retrieval
          -> Reranker
          -> LLMSupport ya termino
          -> VERIFY
          -> LLM
```

Cuando el pipeline llega a VERIFY, el LLMSupport ya dejo preparada una sugerencia.

No agrega practicamente latencia al camino critico. Esto esta dado por diseno: ejecucion paralela en CPU con recursos dedicados, sin competir por GPU con el pipeline.

## 7. Posicion en los planos arquitectonicos

LLMSupport es **transversal como Observability** (ADR-0005), pero con razonamiento LLM.

```
CONTROL          | CAPABILITIES       | KNOWLEDGE
Controller,      | Retrieval, Gen,    | Knowledge System
Policy Engine,   | Assess, Verify,    | (Warm Artifacts)
Registry         | Planner, Tools
                 |
-----------------+--------------------+------------------------
transversal: OBSERVABILITY
transversal: EVALUATION
transversal: CONFIGURATION
transversal: LLMSUPPORT (observador paralelo)
```

- No es capability (no se invoca via Registry).
- No es policy (no decide).
- No es evaluation (no produce senales duras).
- Es un **observador** que consume eventos del `TraceSink` y produce hipotesis.

## 8. Integracion con arquitectura existente

| Componente | Relacion con LLMSupport |
|---|---|
| `TraceSink` (ADR-0005) | Fuente de eventos. LLMSupport se suscribe a `TraceEvent` del pipeline. `TraceSink.emit()` ya es push-based; no requiere modificar ADR-0005, solo un sink de fan-out. |
| `ExecutionState` (ADR-0004) | LLMSupport lee estado (read-only). Nunca escribe. |
| `PolicyEngine` (ADR-0013) | Consumidor opcional de hipotesis. Policy decide si las usa. |
| `CompositionRoot` (ADR-0014) | Cablea el LLMSupport (P13 — inyeccion de dependencias). |
| `ModelProvider` (ADR-0007) | LLMSupport usa un **ModelProvider dedicado** con un modelo pequeno (ver 8.1). |

### 8.1 Modelo dedicado pequeno en CPU

El LLMSupport **no usa el mismo modelo** que la capability de generation del pipeline. Usa un **modelo dedicado mas pequeno** (ej. 3B) que corre **paralelamente en CPU**.

```
Pipeline (GPU si disponible):
  LLM principal (ej. 8B)
  -> generation, verify, etc.

LLMSupport (CPU):
  Modelo pequeno (ej. 3B)
  -> observacion paralela, hipotesis
```

Razones arquitectonicas:

- **No compite por recursos GPU** — el modelo principal del pipeline tiene la GPU dedicada. El LLMSupport corre en CPU sin interferir.
- **Modelo mas liviano es suficiente** — el LLMSupport no genera respuestas; solo razona sobre el estado del pipeline y produce hipotesis. Un modelo 3B es adecuado para esta tarea.
- **Latencia marginal** — un modelo 3B en CPU puede producir una hipotesis en tiempos compatibles con la duracion de las etapas del pipeline (retrieval, reranker, verify).
- **ModelProvider inyectado** (P13) — el LLMSupport recibe su propio `ModelProvider` configurado para el modelo pequeno. El contrato (ADR-0007) se preserva; solo cambia la implementacion cableada en Composition Root.

Configuracion en Composition Root:

```python
# Ejemplo conceptual (no es codigo de implementacion)
llm_support_provider = OllamaProvider(model="qwen2.5:3b", device="cpu")
llm_support = LLMSupport(
    model_provider=llm_support_provider,
    trace_sink=bundle.trace_sink,
    mode="passive",
)
```

- `llm_support.model` = modelo pequeno dedicado (ej. 3B)
- `llm_support.device` = CPU (no compite con GPU del pipeline)
- `llm_support.mode` = `"passive"` | `"advisory"` | `"off"`
- El modelo es reemplazable (P2 — contratos estables, implementaciones desechables)

## 9. Fase 1: Observabilidad pura (pasivo)

```
Pipeline normal
  +
LLMSupport
  |
  v
Log: "Yo hubiera hecho retry."
```

Nada cambia. El pipeline ejecuta exactamente igual.

Se mide:
- Precision de hipotesis: cuando LLMSupport sugiere retry, era necesario?
- Recall de hipotesis: cuando el pipeline fallo, LLMSupport lo detecto?
- Comparacion con decisiones reales del Policy Engine

Despues se descubre:

> "El LLMSupport detecto correctamente el 82% de los retrieval malos."

Recien ahi se habilita influencia.

Esto sigue perfectamente el principio **P17**: la observabilidad no cambia el comportamiento.

## 10. Fase 2: Influencia opt-in via Policy Engine

Cuando los benchmarks validan la utilidad del LLMSupport:

```
LLMSupport
  |
  produce: Hypothesis { suggestion: RETRY_RETRIEVAL, confidence: 0.71 }
  |
  v
Policy Engine
  |
  v  Decide:
     - Ignorar?
     - Aceptar?
     - Pedir retry?
     - Cambiar estrategia?
```

El LLM nunca toma la decision. Produce una opinion. El Policy Engine conserva ownership (P16).

Habilitacion gradual por config (Composition Root):
- `llm_support.mode = "passive"` (Fase 1 — solo log)
- `llm_support.mode = "advisory"` (Fase 2 — hipotesis disponibles para Policy Engine)
- `llm_support.mode = "off"` (default seguro)

**Nota de transicion arquitectonica:** el modo `advisory` cruza la frontera de P17 — el LLMSupport deja de ser observabilidad pura y pasa a influir el control. Habilitarlo exige un ADR propio y medir previamente la Fase 1.

## 11. Ownership y fronteras

| Puede LLMSupport | No puede LLMSupport |
|---|---|
| Leer `ExecutionState` (read-only) | Invocar capabilities |
| Suscribirse a `TraceEvent` | Modificar `ExecutionState` |
| Producir `Hypothesis` | Sobrescribir decisiones del Policy Engine |
| Loggear hipotesis | Bloquear el pipeline |
| Correr en paralelo sin bloquear | Reemplazar ASSESS o VERIFY |
| Terminar antes que el pipeline | Escribir en `signals` o `last_decision` |

## 12. Comparativa: ASSESS/VERIFY vs LLMSupport

| Aspecto | ASSESS / VERIFY | LLMSupport |
|---|---|---|
| Momento | Discreto, post-hoc | Continuo, proactivo |
| Tipo de output | `EvaluationSignal` (pass/fail) | `Hypothesis` (sugerencia + confidence) |
| Gatea? | Si (hard gate o soft signal) | No |
| Bloquea? | Si (el pipeline espera el resultado) | No (corre en paralelo) |
| Decide? | No (produce senales) | No (produce opiniones) |
| Usa LLM? | No (determinista, local-first) | Si (razonamiento LLM) |
| Consume eventos? | No (evalua estado en un punto) | Si (reactivo, incremental) |
| Owner de la decision | Policy Engine interpreta la senal | Policy Engine interpreta la hipotesis |

ASSESS y VERIFY son evaluadores discretos que gatean en momentos especificos. LLMSupport es un observador continuo que razona sobre el estado global del run. Son complementarios, no redundantes.

---

## 13. Riesgos

| Riesgo | Probabilidad | Impacto | Mitigacion |
|---|---|---|---|
| LLMSupport produce hipotesis ruidosas | Media | Medio | Fase 1 pasiva para medir precision/recall antes de habilitar influencia |
| LLMSupport acopla al Consumer a un modelo | Media | Alto | ModelProvider inyectado (P13); modelo configurable |
| Policy Engine se vuelve dependiente de LLMSupport | Media | Alto | Policy Engine funciona sin LLMSupport; hipotesis son input opcional |
| Modo advisory habilitado sin evidencia | Media | Alto | ADR propio obligatorio + gate de precision/recall (P4, P16) |

---

## 14. Open questions

1. **Presupuesto del LLMSupport**: como acotar llamadas LLM del observador (max_hypothesis_updates?)
2. **Serializacion de hipotesis**: van en `ExecutionState.metadata`? En trazas? En un campo dedicado?
3. **Relacion LLMSupport con Planner**: el Planner planea antes; LLMSupport observa durante. Como coordinan?
4. **LLMSupport y streaming**: si el LLM esta streamando tokens, puede LLMSupport observar el stream?
5. **Multiple hipotesis**: puede LLMSupport mantener multiples hipotesis simultaneas?
6. **Aprendizaje del LLMSupport**: puede la precision de hipotesis mejorar con el tiempo?
7. **Fan-out del TraceSink**: disposicion del sink compuesto (orden de sinks, aislamiento de errores entre sinks)

---

## 15. Takeaways

1. **LLMSupport es un observador paralelo, no un decisor.** Produce hipotesis; el Policy Engine decide.
2. **LLMSupport empieza 100% pasivo.** Observabilidad pura, medir, luego habilitar influencia (P17).
3. **LLMSupport no bloquea ni agrega latencia.** Corre en paralelo, reactivo a eventos, con recursos dedicados en CPU.
4. **LLMSupport es complementario a ASSESS/VERIFY.** ASSESS/VERIFY evaluan discreto; LLMSupport observa continuo.
5. **El Policy Engine conserva ownership.** LLMSupport alimenta; Policy Engine decide (P16).
6. **El modo advisory cruza P17** y exige ADR propio; no es solo un flag de config.
7. **No depende del contrato Warm.** Implementable de forma independiente al split Builder/Consumer.
8. **No se implementa ahora.** Este research prepara la promocion futura a ADR.

---

## 16. Criterio de promocion a ADR

Este research puede promoverse a ADR cuando se acuerde al menos:

- LLMSupport como cuarto plano transversal (junto a Observability, Evaluation, Configuration)
- contrato de `Hypothesis` (suggestion, confidence, reasoning, stage, run_id) — distinto de `EvaluationSignal` y de `ActionDecision`
- modelo reactivo incremental basado en eventos (`TraceSink.emit`, push-based)
- principio de no-bloqueo (paralelo, sin latencia en camino critico)
- modelo dedicado pequeno en CPU via ModelProvider inyectado (P13)
- Fase 1 pasiva obligatoria antes de Fase 2 advisory
- ownership: Policy Engine decide, LLMSupport opina (P16)
- habilitacion gradual por config (off -> passive -> advisory)
- ADR separado para el modo advisory (cruce de P17)
- presupuesto del LLMSupport acotado

Hasta entonces permanece como research de arquitectura de largo plazo.
