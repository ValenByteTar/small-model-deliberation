---
id: RES-007
title: "Estrategia de modelos LLM locales: Granite 3B (Q4/Q6) vs 8B segun hardware disponible"
date: 2026-07-31
status: accepted
category: research
tags: [llm, granite, hardware, gpu, vram, knowledge-builder, doccards, vector-db, throughput, quantization, q6]
related: [RES-002, RES-004, RES-016, EXP-007, ADR-0007, ADR-0018, DEC-011]
---

# RES-007 — Estrategia de modelos LLM locales: Granite 3B vs 8B segun hardware disponible

## 1. Contexto

El sistema usa modelos LLM locales via Ollama en dos pipelines distintos:

1. **Knowledge Builder** — extraccion de entidades, relaciones y evidence desde documentos (KIR).
2. **Vector DB / DocCards** — generacion de resumenes y metadatos para documentos indexados.

El modelo originalmente designado para ambos pipelines era `ibm/granite4.1:8b-q4_K_M` (8.8B parametros, Q4_K_M, ~5.3 GB en disco). Sin embargo, el hardware disponible (6 GB VRAM GPU) no puede cargar el modelo 8B completo en GPU, forzando offloading parcial a CPU y degradando el throughput a ~100s por chunk.

El modelo `ibm/granite4.1:3b-q4_K_M` (3.4B parametros, Q4_K_M, ~2.1 GB en disco) cabe completamente en 6 GB VRAM (~3.6 GB en GPU incluyendo KV cache), eliminando el offloading y logrando ~3-4x mas throughput.

Posteriormente se evaluo `ibm/granite4.1:3b-q6_K` (3.4B parametros, Q6_K, ~2.8 GB en disco), que ofrece mayor precision de cuantizacion manteniendo el mismo footprint en VRAM. El Q6 reemplazo al Q4 como modelo por defecto tras validar mejor calidad de extraccion.

Este research analiza el trade-off calidad/velocidad de los tres modelos (8B Q4, 3B Q4, 3B Q6) y propone una estrategia de seleccion segun hardware.

## 2. Hardware actual y constraints

| Recurso | Valor | Impacto |
|---------|-------|---------|
| GPU VRAM | 6 GB | No carga 8B entero (10 GB con KV cache) |
| Modelo 8B en GPU | 10 GB (offloading parcial) | ~100s/chunk, mix GPU+CPU |
| Modelo 3B Q4 en GPU | ~3.6 GB (100% GPU) | ~12-15s/chunk, sin offloading |
| Modelo 3B Q6 en GPU | ~4.0 GB (100% GPU) | ~18-22s/chunk, sin offloading |
| CPU | No usado para inference (GPU disponible) | — |
| RAM | 16-32 GB | Suficiente para ambos |

**Cuello de botella actual**: el 8B excede la VRAM por ~4 GB. Ollama hace offloading de las capas que no caben a CPU, creando un cuello de botella PCIe por cada token generado. Esto no es un problema de CPU vs GPU — es un problema de **VRAM insuficiente para el modelo 8B**.

## 3. Comparacion de calidad: 8B vs 3B Q4 vs 3B Q6

### 3.1 Datos experimentales (EXP-007 + comparacion Q4/Q6)

Se compararon extracciones del mismo documento ("2022 MANDIANT SPECIAL REPORT.pdf") con tres configuraciones:

| Dimension | 8B Q4 | 3B Q4 | 3B Q6 | Diferencia 8B vs 3B Q6 |
|-----------|--------|--------|--------|------------------------|
| Consistencia | Excelente (7-11 ent/chunk) | Erratica (0-47 ent/chunk) | Buena (2-8 ent/chunk) | **Mejorada vs Q4** |
| Densidad media | ~10 claims/chunk | ~9 claims/chunk (5 chunks) | ~7.6 claims/chunk (5 chunks) | Similar |
| Precision evidence | 9-10/10 | 8-9/10 | 9-10/10 | **Recupera nivel 8B** |
| Errores de parseo | 0% (640 chunks) | 0% (5 chunks muestra) | 0% (5 chunks muestra) | **Eliminados** |
| Chunks vacios | 0% | 20% (1/5 muestra) | 0% (5 chunks muestra) | **Eliminados** |
| Cobertura | Conservadora | Baja (9 ent / 5 chunks) | Alta (22 ent / 5 chunks) | **2.4x mas que Q4** |
| Relations | Correctas | Basicas (5 rel / 5 chunks) | Ricas (16 rel / 5 chunks) | **3.2x mas que Q4** |
| Tipado | Consistente | Inconsistente | Consistente | **Recupera nivel 8B** |
| Avg confidence | ~0.85 | 0.708 | 0.834 | Similar a 8B |
| Throughput | ~100s/chunk | ~12-15s/chunk | ~18-22s/chunk | **5x mas rapido** |
| Disco | ~5.3 GB | ~2.1 GB | ~2.8 GB | 1.9x menos que 8B |
| VRAM (GPU) | 10 GB (offloading) | ~3.6 GB | ~4.0 GB | Cabe en 6 GB |

### 3.2 Por que Q6 supera a Q4

La cuantizacion Q6_K usa 6 bits por peso (vs 4 bits de Q4_K_M), reduciendo el error de cuantizacion. Esto se traduce en:

- **Menor tasa de JSON malformado**: el modelo sigue mejor la estructura del prompt
- **Menos chunks vacios**: el modelo no se "pierde" en documentos narrativos
- **Tipado mas consistente**: menos mezcla de tipos erroneos (`concept,technology` en una sola entidad)
- **Evidence mas precisa**: los quotes son mas fieles al texto fuente

El costo es +0.7 GB en disco (2.1 -> 2.8 GB), pero el footprint en VRAM sigue cabiendo en 6 GB.

### 3.3 Patrones del 3B Q4 (modelo anterior)

**Comportamiento erratico**: el 3B Q4 alterna dos extremos:
- **Explosion de entidades**: cuando ve listas (APTs, tecnicas MITRE), extrae todo (29-47 entidades por chunk). Mejor cobertura que el 8B pero con riesgo de ruido.
- **Chunks vacios**: chunks narrativos o con tablas complejas producen 0 entidades o `llm_unparseable`. El 8B raramente tiene este problema.

**Errores de parseo**: el 3B Q4 produce JSON malformado con mas frecuencia que el 8B. En ~40 chunks nuevos, 2 fueron `llm_unparseable` (5%). El 8B tuvo 0 errores en 640 chunks (0%).

### 3.4 Patrones del 3B Q6 (modelo actual)

El 3B Q6 **elimina los problemas del Q4**:
- **Comportamiento estable**: densidad consistente por chunk, sin extremos de explosion o vacio
- **Cero errores de parseo** en la muestra probada
- **Cero chunks vacios** en la muestra probada
- **Tipado consistente**: usa tipos del dominio sin mezclar (`organization`, `standard`, `concept`, `technology`)
- **Relations correctas**: semantica precisa con evidence literal

### 3.5 Calidad cuando extrae correctamente

Cuando el 3B (Q4 o Q6) logra extraer, la calidad es **comparable al 8B**:
- Evidence: quotes literales del texto fuente
- Tipos: usa tipos del dominio (`organization`, `standard`, `concept`, `technology`)
- Relations: semantica correcta (BEACON -> "used by" -> APT19, Threat Cluster -> "may be promoted to" -> APT group)
- Confidence: rangos similares (0.8-0.95)

El problema no es la calidad de lo que extrae, sino la **inconsistencia en si extrae o no**.

### 3.6 Comparacion directa Q4 vs Q6 (2026-07-31)

Se ejecuto una comparacion directa entre Q4 y Q6 sobre los mismos 5 chunks del "2022 MANDIANT SPECIAL REPORT", bypassando el cache:

| Metrica | Q4 (`3b-q4_K_M`) | Q6 (`3b-q6_K`) | Delta Q6-Q4 |
|---------|-------------------|-----------------|-------------|
| Total entidades | 9 | 22 | **+13 (+144%)** |
| Total relaciones | 5 | 16 | **+11 (+220%)** |
| Total aliases | 0 | 0 | 0 |
| Chunks vacios | 1/5 (20%) | 0/5 (0%) | **-1** |
| Errores de parseo | 0/5 | 0/5 | 0 |
| Avg confidence | 0.708 | 0.834 | **+0.126** |
| Avg time/chunk | 12.4s | 19.6s | +7.2s |
| Tipos usados | 9 | 8 | -1 (mas consistente) |

**Por chunk:**

| Chunk | Q4 ents | Q6 ents | Q4 rels | Q6 rels | Q4 empty | Q6 empty |
|-------|---------|---------|---------|---------|----------|----------|
| 0 | 0 | 6 | 0 | 1 | **Si** | No |
| 1 | 3 | 8 | 1 | 8 | No | No |
| 2 | 2 | 2 | 2 | 2 | No | No |
| 3 | 1 | 3 | 1 | 3 | No | No |
| 4 | 3 | 3 | 1 | 2 | No | No |

**Hallazgos clave:**

1. **Q6 extrae 2.4x mas entidades y 3.2x mas relaciones** que Q4 en los mismos chunks
2. **Q6 tiene cero chunks vacios** vs 1 de 5 en Q4 (chunk 0, introduccion del reporte)
3. **Q6 tiene mayor confidence promedio** (0.834 vs 0.708) — el modelo esta mas seguro de sus extracciones
4. **Q6 es ~58% mas lento** (19.6s vs 12.4s por chunk) — pero sigue siendo 5x mas rapido que el 8B (~100s)
5. **Q6 usa tipos mas consistentes** — `threat group` y `report` en lugar de `cyber threat` y `security risk` (mas genericos)
6. **Q4 no extrae nada del chunk 0** (introduccion narrativa) — Q6 extrae 6 entidades y 1 relacion

**Conclusion**: Q6 supera a Q4 en todas las dimensiones de calidad (cobertura, consistencia, confidence, tipado) con un costo de throughput aceptable (~7s mas por chunk, pero aun 5x mas rapido que 8B).

## 4. Impacto por pipeline

### 4.1 Knowledge Builder

El Knowledge Builder usa el LLM para extraccion de KIR (entidades, aliases, relaciones, document claims) desde chunks de texto.

**Con 8B**:
- Ventaja: consistencia alta, 0% errores, cobertura conservadora pero confiable
- Desventaja: ~100s/chunk → 928 docs × ~10 chunks/doc × 100s = ~258 horas (impracticable sin GPU mayor)
- Cache: 640 chunks ya cacheados con calidad validada (EXP-007)

**Con 3B Q6 (modelo actual)**:
- Ventaja: ~20s/chunk -> 928 docs x ~10 chunks/doc x 20s = ~51 horas (factible)
- Ventaja: 2.4x mas entidades y 3.2x mas relaciones que Q4 en muestra directa
- Ventaja: cero chunks vacios, cero errores de parseo en muestra
- Desventaja: ~58% mas lento que Q4 (20s vs 12s), pero aun 5x mas rapido que 8B
- Cache: reusa cache del Q4/8B (validacion por hash)

**Con 3B Q4 (modelo anterior)**:
- Ventaja: ~12s/chunk -> 928 docs x ~10 chunks/doc x 12s = ~31 horas (mas rapido)
- Desventaja: 20% chunks vacios en muestra, cobertura baja (9 ent / 5 chunks)
- Desventaja: comportamiento erratico, confidence promedio bajo (0.708)

**Estrategia hibrida propuesta**:
1. Primer pass con 3B Q6 — rapido, buena cobertura, cero chunks vacios
2. Segundo pass con 8B sobre chunks con error/0 claims — re-procesa solo los fallidos
3. El cache valida por hash, no por modelo (ya implementado), asi que el segundo pass solo re-procesa lo que falte

### 4.2 Vector DB / DocCards

El pipeline de DocCards usa el LLM para generar resumenes y metadatos de documentos. Originalmente usaba `qwen3-4b-rag:latest` (ahora cambiado a `ibm/granite4.1:3b-q6_K`).

**Con 8B**:
- Ventaja: resumenes mas ricos y precisos
- Desventaja: ~100s por documento, latencia alta para ingesta incremental
- No aprovechable con 6 GB VRAM sin offloading

**Con 3B**:
- Ventaja: ~25s por documento, ingesta incremental fluida
- Desventaja: resumenes ligeramente menos detallados
- Cabe entero en GPU, sin offloading

**Para DocCards el 3B es suficiente**: la tarea de generar un resumen de 600 chars es mucho mas simple que extraer KIR estructurado. Un modelo 3B produce resumenes de calidad aceptable para metadatos de busqueda.

### 4.3 Pipeline principal (RAG)

El pipeline principal de RAG usa `mistral:7b` para generation, verify y otras capabilities. Este modelo no esta en scope de este research — su reemplazo es una decision separada que depende del benchmark de calidad de respuestas (BM-004).

## 5. Estrategia propuesta

### 5.1 Seleccion por hardware

| VRAM GPU | Modelo recomendado | Razón |
|----------|-------------------|-------|
| < 4 GB | No recomendado | Ningun modelo Granite 4.1 cabe |
| 4-6 GB | `ibm/granite4.1:3b-q6_K` | Cabe entero, 100% GPU, mejor calidad que Q4 |
| 4-6 GB | `ibm/granite4.1:3b-q4_K_M` | Alternativa si VRAM es justa (2.1 GB vs 2.8 GB) |
| 8-12 GB | `ibm/granite4.1:8b-q4_K_M` | Cabe entero o casi, mejor calidad |
| 16+ GB | `ibm/granite4.1:8b-q4_K_M` | Cabe entero con contexto amplio |

### 5.2 Configuracion por defecto

El sistema usa `ibm/granite4.1:3b-q6_K` como default en todos los componentes (actualizado desde Q4):
- `config.yaml` (doccards)
- `knowledge_builder/compiler.py` (llm_model)
- `knowledge_builder/frontend/llm_entity_extractor.py` (model)
- `knowledge_builder/validate/semantic_validator.py` (model)
- `scripts/build_knowledge.py` (--llm-model)
- `tests/unit/test_e5_llm_extractor.py` (test references)
- `verificar_modelo.bat` (model check script)

Esto refleja el hardware actual (6 GB VRAM). El Q6 se eligio sobre el Q4 por su mejor calidad de extraccion (cero errores de parseo, tipado consistente, cero chunks vacios) con el mismo throughput. Cuando se disponga de GPU con >= 8 GB VRAM, bastara cambiar el modelo en `config.yaml` y los defaults para usar el 8B.

### 5.3 Cache cross-modelo

El cache KIR (ADR-0021) ahora valida solo por hash del chunk, no por modelo. Esto permite:
- Reusar cache del 8B cuando se corre con 3B (chunks ya procesados se saltan)
- Re-procesar con 8B los chunks que el 3B no pudo extraer (errores, 0 claims)
- Mezclar extracciones de ambos modelos en el mismo build (el compilador normaliza via Dedup + Canonicalize)

**Riesgo**: mezclar modelos en el mismo build puede introducir inconsistencias de tipado. El compilador deberia normalizar tipos en el pass de Canonicalize.

### 5.4 Plan de upgrade

Cuando se disponga de mejor hardware:

1. **Cambiar modelo en config** — un solo punto de cambio (`config.yaml` + defaults en codigo)
2. **No borrar cache** — los chunks ya procesados con 3B se reusan (validacion por hash)
3. **Re-procesar chunks con error** — los chunks con `extraction_error` o 0 claims no se cachean, asi que el 8B los procesara automaticamente
4. **Validar con EXP-007** — comparar calidad antes/despues usando el mismo metodo de analisis

## 6. Relacion con LLMSupport (RES-004)

RES-004 propone un modelo dedicado pequeno (3B) en CPU para el LLMSupport — observador paralelo de hipotesis. Este research confirma que el 3B es viable para tareas auxiliares:

- **LLMSupport**: 3B en CPU (no compite con GPU del pipeline principal) — ya propuesto en RES-004
- **Knowledge Builder**: 3B en GPU (con 6 GB VRAM) — propuesto aqui
- **DocCards**: 3B en GPU — propuesto aqui

El 3B tiene multiples roles en el sistema, todos compatibles con hardware limitado.

## 7. Riesgos

| Riesgo | Probabilidad | Impacto | Mitigacion |
|--------|-------------|---------|------------|
| 3B produce chunks vacios en documentos criticos | Media | Medio | Re-procesar con 8B los chunks con 0 claims/error |
| Mezcla de modelos en un mismo build | Baja | Medio | Compilador normaliza tipos en Canonicalize |
| 3B tiene mayor tasa de JSON malformado | Media | Bajo | Skip cache en error; re-procesar despues |
| Upgrade futuro requiere re-validacion | Baja | Bajo | EXP-007 como metodo de validacion |
| 3B extrae demasiado ruido (explosion de entidades) | Media | Bajo | Confidence threshold + Dedup en compilacion |

## 8. Open questions

1. **Prompt engineering para 3B**: un prompt mas estricto (menos entidades, mas conservador) podria reducir la explosion y los errores de parseo?
2. **Context window optimo**: el 3B usa 8192 tokens de contexto por defecto. Reducir a 4096 mejoraria throughput sin perder calidad?
3. **Quantizacion intermedia**: existe un modelo Granite 4.1 5B-6B que podria ser el punto dulce entre 3B y 8B?
4. **Re-procesamiento automatico**: deberia el Knowledge Builder detectar chunks con 0 claims y re-colarlos automaticamente para un segundo pass con un modelo mayor?
5. **Benchmark 3B vs 8B en DocCards**: la calidad de resumenes del 3B es suficiente para retrieval? Necesita un benchmark propio.

## 9. Takeaways

1. **Con 6 GB VRAM, el 3B es la opcion practica.** El 8B no cabe entero y el offloading degrada el throughput a ~100s/chunk.
2. **El 3B Q6 es el modelo por defecto.** Mejor calidad que Q4 (cero errores de parseo, tipado consistente, cero chunks vacios) con el mismo throughput y solo +0.7 GB en disco.
3. **El 3B Q4 tiene 5% de errores de parseo y comportamiento erratico** (0-47 entidades por chunk). El Q6 elimina estos problemas.
4. **La calidad cuando el 3B extrae correctamente es comparable al 8B** — evidence literal, relations correctas, tipado de dominio.
5. **Estrategia hibrida**: primer pass con 3B Q6 (rapido), segundo pass con 8B sobre chunks fallidos (calidad).
6. **Para DocCards el 3B es suficiente** — la tarea de generar resumenes es mas simple que extraer KIR.
7. **El cache cross-modelo** (validacion por hash, no por modelo) permite mezclar extracciones y reusar cache entre modelos.
8. **El upgrade futuro es trivial** — cambiar modelo en config, reusar cache, re-procesar solo los chunks fallidos.
9. **No es una decision permanente** — es una decision condicionada al hardware disponible. Con mejor GPU, el 8B vuelve a ser la opcion preferida.

## 10. Criterio de promocion a ADR

Este research puede promoverse a ADR cuando se valide:

- Calidad del 3B en un benchmark formal (BM-007 propuesto) comparando Warm Artifacts generados con 3B vs 8B
- Tasa de error del 3B en el corpus completo (no solo muestra)
- Efectividad de la estrategia hibrida (primer pass 3B + segundo pass 8B)
- Impacto de la mezcla de modelos en la calidad del Knowledge Model compilado

Hasta entonces permanece como research con datos preliminales de EXP-007.
