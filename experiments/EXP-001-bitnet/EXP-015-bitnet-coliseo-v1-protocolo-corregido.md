---
id: EXP-015
title: "BitNet b1.58-2B-4T en Coliseo v1: confirmacion de incapacidad semantica (55 casos, protocolo corregido)"
date: 2026-08-16
status: completed
category: experiment
components: [bitnet_provider, semantic_ensemble, ollama_instances]
tags: [bitnet, semantic-evaluation, benchmark-v2, cpu, ensemble, coliseo, protocol-corrected, capacity-limit]
related: [PM-003, EXP-010, POST-001, EXP-012, PAT-001]
supersedes: EXP-010
superseded_by: null
---

# EXP-015 - BitNet b1.58-2B-4T en Coliseo v1 (protocolo corregido)

## Hipotesis

BitNet-b1.58-2B-4T fue condenado en PM-003/EXP-010 basado en 12 casos
con `num_predict=10` (ensemble) y `max_tokens=256` (single). Dado que
POST-001 descubrio que el protocolo `num_predict=10` + parser leniento
contamino los resultados de Llama3.2 (16.4% → 58.2%), es necesario
verificar si BitNet sufrio contaminacion similar.

**Hipotesis primaria**: BitNet en el benchmark v2 completo (55 casos)
con protocolo corregido produce resultados significativamente mejores
que el historico de 12 casos.

**Hipotesis secundaria**: El ensemble de 4 workers supera al single
cuando se evalua sobre 55 casos con categorias diagnosticas.

## Motivacion

EXP-010 uso un dataset de 12 casos (3 SUPPORTS, 4 CONTRADICTS, 3
UNRELATED, 2 PARTIAL) — estadisticamente debil. Ademas, el experimento
de ensemble uso `num_predict=10` que truncaba las respuestas. BitNet
nunca fue probado en el benchmark v2 de 55 casos con 10 categorias
diagnosticas (adversarial, direct_evidence, explicit_contradiction,
etc.).

## Configuracion

- **Modelo**: BitNet-b1.58-2B-4T (ggml-model-i2_s.gguf, 2B ternario)
- **Backend**: llama-server (bitnet.cpp), CPU, 4 instancias paralelas
- **Instancias**: 4, puertos 8081-8084, 1 thread c/u
- **Benchmark**: semantic_assessment_v2.json (55 casos, 10 categorias)
- **Protocolo**:
  - max_tokens=128 (presupuesto ampliado, no truncado)
  - temperature=0.0
  - Parser estricto (no defaultea a UNRELATED)
  - JSON via instruccion de prompt (llama-server no soporta
    format=json_schema)
  - raw conservado por caso para auditoria
- **Configs**:
  - single (neutral)
  - ensemble_2 (entailment + skeptical)
  - ensemble_4 (entailment + skeptical + contradiction + neutral)
- **Ejecucion**: Fase 1 (single + ensemble_2 en paralelo, 3 instancias),
  Fase 2 (ensemble_4 solo, 4 instancias)
- **Hardware**: CPU (8 cores logicos), 4 instancias BitNet + microcoliseum
  Ollama GPU en paralelo

## Resultados

### Accuracy general

| Config | Historico (12 casos) | Controlado (55 casos) | Delta |
|--------|---------------------|----------------------|-------|
| single | 33.3% | **29.1%** (16/55) | -4.2% |
| ensemble_2 | 33.3% | **36.4%** (20/55) | +3.1% |
| ensemble_4 | 41.7% | **30.9%** (17/55) | -10.8% |

### Errores por categoria (single)

| Categoria | Correctos/Total | Observacion |
|-----------|----------------|-------------|
| adversarial | 2/6 | solo acierta casos obvios |
| direct_evidence | 0/6 | **no acierta ni uno** |
| explicit_contradiction | 4/5 | unico punto fuerte |
| implicit_contradiction | 3/5 | moderado |
| negation | 4/6 | moderado |
| over_specificity | 1/5 | falla casi todos |
| paraphrase | 0/6 | **no acierta ni uno** |
| partial_support | 1/6 | falla casi todos |
| wrong_context | 0/5 | **no acierta ni uno** |
| wrong_subject | 1/5 | falla casi todos |

### Latencia

| Config | Avg lat | Wall time |
|--------|---------|-----------|
| single | 38.1s/caso | 2097s (35 min) |
| ensemble_2 | 64.0s/caso | 2049s (34 min, paralelo con single) |
| ensemble_4 | 177.3s/caso | 2621s (44 min) |

La latencia de ensemble_4 (177s/caso) es altisima debido a
oversubscription de CPU: 4 instancias BitNet + microcoliseum Ollama
GPU = ~620% CPU en un sistema de 8 cores logicos.

### Protocol validity

100% en todas las configs. BitNet sigue el formato (produce una
relacion valida del vocabulario) pero no comprende la semantica.

## Comparacion con otros modelos (benchmark v2, 55 casos)

| Modelo | single | ensemble_2 | ensemble_4 |
|--------|--------|------------|------------|
| Qwen3 4B | 78.2% | 83.6% | 81.8% |
| Granite 3B | 61.8% | 67.3% | 76.4% |
| Llama3.2 3B (corregido) | 58.2% | 58.2% | 63.6% |
| **BitNet 2B** | **29.1%** | **36.4%** | **30.9%** |

BitNet esta ~30 puntos por debajo de Llama3.2 (el siguiente peor) y
~50 puntos por debajo de Qwen3. La diferencia es abismal.

## Conclusion

**Hipotesis primaria refutada.** BitNet con protocolo corregido en 55
casos NO produce resultados significativamente mejores que el
historico. El single baja ligeramente (33.3% → 29.1%) y el ensemble_4
baja significativamente (41.7% → 30.9%). El historico de 12 casos
sobreestimaba la capacidad de BitNet, no la subestimaba.

**Hipotesis secundaria refutada.** El ensemble_4 (30.9%) no supera al
ensemble_2 (36.4%). El patron de EXP-010 se confirma: el ensemble no
compensa la limitacion de capacidad fundamental del modelo.

**PM-003 se sostiene.** La condena de BitNet-b1.58-2B-4T como
semanticamente insuficiente es correcta. A diferencia de Llama3.2
(POST-001), BitNet no fue contaminado por el protocolo — su problema
es de capacidad real, no de configuracion experimental.

### Evidencia clave

1. **direct_evidence 0/6**: BitNet no reconoce soporte directo
   (claim-evidence obvios). Esta es la categoria mas facil del
   benchmark.
2. **paraphrase 0/6**: no reconoce parafrasis.
3. **wrong_context 0/5**: no detecta cuando el contexto es incorrecto.
4. **explicit_contradiction 4/5**: unico punto fuerte — detecta
   contradicciones explicitas (negacion literal).
5. **protocol_validity 100%**: sigue el formato pero no comprende.

### Diferencia con Llama3.2 (POST-001)

Llama3.2 fue contaminado porque:
- Sus respuestas eran largas y verbosas ("Based on the analysis...")
- `num_predict=10` truncaba antes de la clasificacion
- El parser leniento defaulteaba a UNRELATED

BitNet NO fue contaminado porque:
- El semantic_pilot uso `max_tokens=256` (sin truncamiento)
- Sus respuestas son cortas y directas (no verbosas)
- El parser encontraba la relacion en el texto (no necesitaba default)

## Recommendation

- [x] Nothing — PM-003 confirmado, no reintentar BitNet para tareas semanticas
- [ ] Benchmark — congelar como referencia de capacidad minima
- [ ] ADR — ADR-0031 sigue deprecado
- [ ] Decision — no DEC

**Accion tomada**: PM-003 confirmado con datos de 55 casos. BitNet
excluido definitivamente de futuros experimentos semanticos. El
umbral de capacidad minima (RES-007: 7B+) se mantiene.

**Nota metodologica**: Este experimento demuestra el valor de
re-evaluar modelos condenados con protocolos corregidos. En el caso
de Llama3.2, la re-evaluacion refuto la condena (16.4% → 58.2%). En
el caso de BitNet, la confirmo (33.3% → 29.1%). Ambos outcomes son
validos — lo importante es que la evaluacion fue justa.
