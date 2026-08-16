---
id: RES-003
category: research
status: accepted
created: 2026-07-27
updated: 2026-07-28
author: human
components: [kernel, capabilities, rag_hybrid, planner, retrieval, verify, assess]
tags: [architecture, query-time, knowledge-consumer, runtime-evolution, migration, warm-artifacts-consumption, confidence-thresholds]
related: [RES-001, RES-002, RES-004, RES-016, ADR-0005, ADR-0006, ADR-0009, ADR-0012, ADR-0013, ADR-0015, ADR-0016, ADR-0019, ADR-0020, DEC-008, BM-002, BM-003, BM-004]
supersedes: null
superseded_by: null
---

# RES-003 - Knowledge Consumer / evolucion del Agentic RAG runtime

> Extracto de RES-001 (original). RES-001 fue seccionado en tres:
> - **RES-001** — El contrato Warm como centro arquitectonico
> - **RES-002** — Knowledge Builder / Knowledge Compiler
> - **RES-003** — Knowledge Consumer / evolucion del Agentic RAG runtime (este documento)
>
> **Nota de seccionamiento (2026-07-28):** el contenido de LLMSupport fue extraido a
> **RES-004** — LLMSupport: observador paralelo de hipotesis. Las dos preocupaciones tienen
> lifecycles y dependencias independientes: el Consumer depende del contrato (RES-001) y del
> Builder (RES-002); LLMSupport solo depende de `TraceSink` y `ModelProvider`.

## Topic

Evolucion del Consumer (Agentic RAG kernel) en query-time: como consume Warm Artifacts publicados por el Builder (RES-002) via el contrato (RES-001) y como migran sus capabilities existentes a leer el contrato.

## Implementation status

**Implementado — post-megaplan W8 (2026-08-15).** Se incorporaron
contratos execution-local para `QueryIR`, `EvidenceItem`, `EvidenceSet` y
`ContextPackage`; `EvidenceSetCapability`, seleccion determinista con
cobertura/diversidad/budget, y observacion runtime de relaciones como Hot
Artifact con provenance. El flujo tipado se activa mediante
`consumer.typed_evidence` y el fallback legacy queda explicito.

VERIFY ahora expone answer-claim candidates y sus Evidence IDs observados;
Warm resolver reporta build/contract identity y filtra relations por
confidence threshold. Estas salidas no mutan Warm Artifacts ni reemplazan
al Knowledge Builder.

**Pendientes resueltos por el megaplan Consumer Knowledge Independence (W1-W8, 2026-08-15):**

- (1) Endurecer claim-level VERIFY semantico: **RESUELTO** — ADR-0028 (VerificationFailureCode + DecisionResult)
- (2) Policies de suficiencia/seleccion: **RESUELTAS** — ADR-0030 (EvidenceQuality Model), ADR-0025 (EvidenceRequirements)
- (3) Eliminacion gradual de fallbacks legacy: **RESUELTA** — monolito legacy eliminado (rag_hybrid.py 10085→3695 lineas, -63%)
- (4) Fase 7c: Relation Layer consumer side: **RESUELTO** — RelationObservationCapability + WarmArtifactResolver filtra por confidence threshold
- (5) Fase 7d: A/B contrato Warm vs monolito: **RESUELTO** — BM-010: +10.7pp pass rate, +0.106 claim coverage, -0.11 avg violations
- (6) Fase 8: Deprecar conocimiento embebido: **RESUELTO** — ADR-0029, EQUIVALENCES_EMBEDDED_TEXT externalized, kernel.enabled=true default
- (7) Bootstrap strategy (total vs lazy): **RESUELTO** — lazy composition root via _get_kernel_bundle()
- (8) Confidence thresholds por capability: **RESUELTOS** — configurables en config.yaml, consumidos por EvidenceQuality

**A/B BM-010**: 84.0% pass rate (vs 73.3% baseline), 0.933 claim coverage, 0.16 avg violations.
**Monolito**: eliminado por reachability, kernel path es el unico camino soportado.

## Sources

- RES-001: El contrato Warm como centro arquitectonico (Warm Artifacts, Artifact Registry, fronteras)
- RES-002: Knowledge Builder / Knowledge Compiler (compilacion de conocimiento en index-time)
- RES-004: LLMSupport, observador paralelo de hipotesis (componente transversal independiente)
- BM-002: A/B Kernel+VERIFY vs Monolito — brecha de 36.3pp causada enteramente por retrieval
- BM-003: A/B Kernel Fase 6 vs Monolito — sin regresion pero brecha persistente
- BM-004: A/B Kernel Fase 6 + bug fixes — brecha reducida a 27.3pp
- DEC-008: Planner + EntityExpansion tunings — wiring completado, impacto medido en BM-004
- ADR-0005: Observability como substrato transversal
- ADR-0006: Evaluation transversal (offline + online)
- ADR-0009: Memory Port (read-only en kernel)
- ADR-0012: Capability Registry
- ADR-0013: Policy Engine (policies de primera clase)
- ADR-0015: Knowledge System (retrieval + get_entity)
- ADR-0016: Definicion del Kernel
- ADR-0019: Contrato epistemico y VERIFY a nivel de claims
- ADR-0020: Ownership de decisiones y contrato de ejecucion observable
- Monolito: `rag_hybrid.py`, `doc_cards.py`, `equivalences_manager.py`, `conceptual_map.py`, `src/rag/entity_extractor.py`, `retrieval_engine.py`

---

## 1. Motivacion

### 1.1 Sintomas observados en A/B

**BM-002** (Fase 4): 45.5% pass rate vs 81.8% monolito. Brecha de 36.3pp.

**BM-003** (Fase 6): 45.5% pass rate (sin regresion). Planner y entity expansion no mejoraron pass rate porque el conocimiento no llegaba a la query de busqueda como artefacto consumible.

**BM-004** (Fase 6 + bug fixes): 54.5% pass rate. Brecha reducida a 27.3pp. Se cerraron dos gaps de data flow en el Consumer:

1. ~~`EntityExpansionCapability` producia entidades expandidas que no llegaban a la query~~ — **FIXED en BM-004**: `RetrievalCapability` y `TwoStageRetrievalCapability` inyectan `expanded_entities`.
2. `PlannerCapability` produce `candidate_docs`, pero el soft boost (+0.05) sigue siendo debil frente a scores de reranker.
3. ~~Two-stage estaba registrado pero no se activaba automaticamente~~ — **FIXED en BM-004**: `LinearRagPolicy` activa `two_stage_retrieval` en el primer pass cuando hay entidades.
4. Las queries restantes (21, 24, 45, 51, 55) siguen fallando porque el Consumer no dispone de conocimiento de dominio compilado (gazetteer completo, equivalencias, relaciones tipadas, roles ricos).

**Mejora observada en BM-004**: +1 pregunta PASS (Q41), +11.1pp doc hit@K, +0.111 MRR.

**Lectura arquitectonica**: los bug fixes mejoraron el consumo de conocimiento ya disponible. No resolvieron la ausencia de un Knowledge Model compilado de alta calidad. Eso es trabajo de compilacion (RES-002), no de parches en runtime.

### 1.2 El Consumer hoy

El Consumer es el Agentic RAG kernel en query-time.

Responsabilidades:

- planificar la consulta
- resolver Warm Artifacts via Resolution Protocol (ver RES-001)
- ejecutar retrieval / generation / verification
- producir Hot Artifacts temporales
- responder

El Consumer:

- nunca interpreta documentos crudos para descubrir dominio
- nunca genera conocimiento estable de dominio
- nunca ejecuta extraccion semantica de corpus
- nunca reconstruye entidades, aliases o roles desde cero
- nunca publica artifacts
- unicamente consume contratos

Cualquier conocimiento de dominio se compila antes del runtime (RES-002).
El Consumer solamente conoce contratos (RES-001).

---

## 2. Como consume el kernel los Warm Artifacts

| Capability actual | Hoy | Con contrato Warm |
|---|---|---|
| `PlannerCapability` | Detecta tipo de query; roles limitados | Lee `doc_roles` + taxonomy + confidence |
| `EntityExpansionCapability` | `_DEFAULT_ALIASES` + memory | Lee `alias_index` + canonical entities |
| `RetrievalCapability` | Inyecta expanded entities (BM-004) | Ademas usa `entity_index` / retrieval metadata |
| `TwoStageRetrievalCapability` | Activado en primer pass (BM-004), fallback a retrieve | Usa `entity_index` compilado para busqueda dirigida |
| `MemoryReadCapability` | Memoria runtime | Igual — memoria no es conocimiento de corpus |
| `VerifyCapability` | Groundedness de respuesta | Igual — opera sobre evidencia de la query (Hot) |

El Consumer puede seguir produciendo Hot Artifacts (`expanded_entities`, `candidate_docs`, etc.). Eso no viola la frontera: son estado de query, no conocimiento estable.

El Consumer no necesita conocer al Builder.
Solo necesita conocer el contrato.
El Consumer no habla con el Builder. Habla con el Registry (RES-001).

### 2.1 Confidence policies del Consumer

Ejemplos (no prescriptivos de implementacion actual):

- expandir solo aliases con `confidence >= 0.85`
- usar relations en comparison solo si `confidence >= 0.9`
- degradar soft-boost si doc role tiene baja confidence
- preferir entity_index entries high-confidence en two-stage
- loggear/telemetry de claims borderline

Confidence es una senal de decision del Consumer, no solo provenance. Ver RES-001 seccion de Confidence contractual.

---

## 3. Comparativa

| Aspecto | Monolito | Kernel actual (F6 + BM-004) | Consumer con contrato Warm |
|---|---|---|---|
| **Centro del sistema** | Codigo monolitico | Kernel + wiring | Contrato Warm (RES-001) |
| **Momento del conocimiento** | Mezcla index/query | Consume poco conocimiento compilado | Compila en index-time (RES-002), consume en query-time |
| **Entity expansion** | Dict + memory + runtime | Alias limitados, ya inyectados en query | `alias_index` compilado + confidence |
| **Doc roles** | Heuristica + LLM opcional | Soft boost debil | `doc_roles` compilados |
| **Equivalences** | 92 grupos manuales | No integradas como artifact | `alias_index` + relations |
| **Two-stage** | Automatico | Activado en primer pass | Guiado por `entity_index` compilado |
| **Relations** | Hechos/ad-hoc | No tipadas | Triples + catalogo controlado |
| **Confidence** | Implicita / ausente | Ausente como contrato | Primordial en claims Warm + Policy |
| **Evolucion a GraphRAG** | Dificil | Dificil | Natural via Relation Layer |
| **Acoplamiento** | Alto | Medio | Bajo (contrato Warm + Registry) |

---

## 4. Migracion incremental

No es big-bang.

### Fase 7a — Compiler minimo + KIR + Warm Artifacts

- Implementar `knowledge_builder/` con front-end -> KIR -> passes -> validation -> back-end (ver RES-002)
- Knowledge Pass API: NormalizePass, CanonicalizePass
- Layers iniciales: Document + Entity
- Publicar: canonical entities, alias index, doc roles, manifest, confidence minima
- Artifact Registry: publication + resolution protocol, staging -> promote (ver RES-001)
- Consumer resuelve Warm Artifacts en `bootstrap.py` via Resolution Protocol
- `EntityExpansionCapability` lee `alias_index`
- `PlannerCapability` lee `doc_roles`
- Compatibilidad con monolito se mantiene

### Fase 7b — Retrieval Layer

- Ya hecho en BM-004 a nivel Consumer:
  - inyeccion de `expanded_entities` en query
  - activacion de two-stage cuando hay entidades
- Pendiente de compiler (RES-002):
  - `entity_index` rico
  - retrieval metadata
  - two-stage guiado por artifact y no solo fallback

### Fase 7c — Relation Layer + catalogo + evidence validation

- Publicar `entity_relations` como triples del catalogo controlado
- Exigir evidence validation + confidence
- Confidence Policy configurable
- Habilitar comparison/balancing en Consumer leyendo relations (sin redescubrir dominio)
- Planner puede filtrar por thresholds de confidence

### Fase 7d — A/B contrato Warm vs monolito

- Eval con `--kernel` + manifest activo
- Medir pass rate, doc hit, recall, MRR
- Objetivo: paridad o mejora vs monolito (81.8% en muestra actual)
- Registry rollback disponible si regresion

### Fases LLMSupport

El observador paralelo LLMSupport tiene su propia trayectoria (pasivo -> advisory) documentada en **RES-004**. Es independiente de esta migracion.

### Fase 8 — Deprecar conocimiento embebido en runtime

- Si A/B es positivo, retirar conocimiento de dominio hardcodeado del Consumer/monolito
- El monolito queda como facade
- `kernel.enabled=true` por defecto
- El contrato Warm queda como unica fuente de conocimiento estable de dominio

---

## 5. Riesgos

| Riesgo | Probabilidad | Impacto | Mitigacion |
|---|---|---|---|
| Calidad inferior al monolito al inicio | Media | Alto | A/B obligatorio antes de deprecar + rollback en Registry |
| Consumer accede directo a archivos | Media | Alto | Resolution Protocol obligatorio; Registry como unico punto de acceso (RES-001) |
| Consumer vuelve a recompilar dominio | Media | Alto | Frontera explicita + reviews de arquitectura |
| Confundir Hot Artifacts con Warm | Baja | Medio | Hot es estado de query; Warm es contrato (RES-001) |

---

## 6. Open questions

1. **Bootstrap del Consumer**: carga total vs lazy por layer/artifact
2. **Thresholds de confidence por capability**: globales vs especificos
3. **Separacion de repos**: cuando justificar Opcion B? (ver RES-002)

---

## 7. Takeaways

1. **El Consumer solo consume contratos.** No reconstruye conocimiento de dominio en runtime.
2. **BM-004 mejoro el consumo; falta mejorar el conocimiento compilado.** Eso es trabajo del Builder (RES-002).
3. **La migracion es incremental.** Fases 7a-7d + Fase 8, con A/B obligatorio en cada paso.
4. **LLMSupport es preocupacion independiente.** Ver RES-004; no comparte dependencias con esta migracion.
5. **No se implementa ahora.** Este research prepara la promocion futura a ADR.

---

## 8. Criterio de promocion a ADR

Este research puede promoverse a ADR cuando se acuerde al menos:

- evolucion del Consumer para consumir Warm Artifacts via Resolution Protocol
- capabilities existentes migran a leer Warm Artifacts (Planner, EntityExpansion, Retrieval, TwoStage)
- thresholds de confidence por capability como configuracion
- bootstrap del Consumer (carga total vs lazy por layer/artifact)
- frontera explicita Consumer/Registry (nunca archivos directos)

Hasta entonces permanece como research de arquitectura de largo plazo.
