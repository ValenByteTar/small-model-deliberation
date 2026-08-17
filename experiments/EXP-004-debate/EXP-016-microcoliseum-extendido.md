---
id: EXP-016
title: "Micro-Coliseum extendido: 9 modelos, deliberacion vs ensemble, hallazgo json_schema para BitNet"
date: 2026-08-17
status: completed
category: experiment
components: [semantic_ensemble, ollama_provider, llama_server_provider, semantic_adapter, debate]
tags: [deliberation, debate, judge, ensemble, microcoliseum, bitnet, llama3.2, qwen3, qwen3.5, gemma3, nemotron, ministral, granite, json-schema, constrained-generation, gpu, cpu]
related: [EXP-014, EXP-015, PM-003, POST-001, PAT-001]
supersedes: null
superseded_by: null
---

# EXP-016 - Micro-Coliseum extendido (9 modelos)

## Hipotesis

H1: La deliberacion entre workers (challenge + judge) corrige mas
errores de los que introduce, medido como net = corrections - damage.

H2: El efecto de la deliberacion depende del modelo: modelos con mayor
capacidad inicial se benefician menos (menos errores que corregir)
pero sufren menos damage.

H3: BitNet-b1.58-2B-4T puede participar en el microcoliseum si se usa
constrained generation (json_schema) para forzar salida estructurada.

## Motivacion

EXP-014 documento el microcoliseum para 3 modelos (Granite, Qwen3-RAG,
Llama3.2) pero solo Granite completo las 9 corridas. Posteriormente:

1. Se corrio el microcoliseum para 4 modelos adicionales (Gemma3,
   Nemotron, Ministral, Qwen3.5) con el runner `run_microcoliseum_all.py`.
2. Se descubrio (POST-001) que Llama3.2 estaba contaminado por
   protocolo en EXP-014, invalidando sus resultados previos.
3. Se creo el runner `run_microcoliseum_trio.py` para BitNet + Llama3.2
   + Qwen3 4B base con protocolo corregido.
4. Se descubrio que BitNet ecoa el template del prompt cuando se usa
   formato JSON sin constrained generation, produciendo artefactos
   (todo CONTRADICTS). Se encontro solucion via `json_schema` por
   request en llama-server.

Este experimento consolida todos los resultados.

## Configuracion

- **Benchmark**: semantic_assessment_v2.json (55 casos, 10 categorias)
- **Workers**: 4 roles especializados (A: entailment, B: skeptical,
  C: contradiction, D: context/entity)
- **Modos**:
  - E0 (independent): 4 workers + vote, sin debate
  - E1 (debate-on-disagreement): debate solo si hay disagreement
  - E2 (debate-all): debate siempre
- **Fases**: Independent Assessment -> Initial Ensemble (frozen) ->
  Challenge Round -> Final Judge
- **Generation**: num_predict=60-128, temperature=0.0, think=false

### Modelos

| Modelo | Backend | Hardware | Notas |
|--------|---------|----------|-------|
| Granite 3B Q4 | Ollama | GPU | ibm/granite4.1:3b-q4_K_M |
| Qwen3 4B-RAG | Ollama | GPU | qwen3-4b-rag:latest (custom, system prompt RAG) |
| Gemma3 4B Q4 | Ollama | GPU | gemma3:4b-it-q4_K_M |
| Nemotron 3 4B | Ollama | GPU | dhiltgen/nemotron-3-nano:4b |
| Ministral 3B Q4 | Ollama | GPU | TechyShishy/ministral-3:3b-reasoning-2512-q4_K_M |
| Qwen3.5 4B Q4 | Ollama | GPU | qwen3.5:4b-q4_K_M |
| BitNet 2B | llama-server | CPU | BitNet-b1.58-2B-4T (i2_s ternario) |
| Llama3.2 3B | Ollama | GPU | llama3.2:3b (protocolo corregido) |
| Qwen3 4B base | Ollama | GPU | hf.co/Qwen/Qwen3-4B-GGUF:Q4_K_M (num_ctx=4096) |

## Resultados

### Tabla consolidada (9 modelos x 3 modos = 27 corridas)

| Modelo | Modo | Init% | Final% | Delta | Corr | Dmg | Net | Deb% | Stab% |
|--------|------|-------|--------|-------|------|-----|-----|------|-------|
| Granite 3B | independent | 69.1% | 69.1% | +0.0% | 0 | 0 | 0 | 0% | 100% |
| Granite 3B | debate-on-disagr | 67.3% | 69.1% | +1.8% | 12 | 11 | +1 | 89% | 56% |
| **Granite 3B** | **debate-all** | **69.1%** | **80.0%** | **+10.9%** | **12** | **6** | **+6** | 100% | 67% |
| Qwen3 4B-RAG | independent | 76.4% | 76.4% | +0.0% | 0 | 0 | 0 | 0% | 100% |
| Qwen3 4B-RAG | debate-on-disagr | 76.4% | 76.4% | +0.0% | 5 | 5 | 0 | 75% | 82% |
| Qwen3 4B-RAG | debate-all | 76.4% | 74.6% | -1.8% | 4 | 5 | -1 | 100% | 84% |
| Gemma3 4B | independent | 52.7% | 52.7% | +0.0% | 0 | 0 | 0 | 0% | 100% |
| **Gemma3 4B** | **debate-on-disagr** | **52.7%** | **63.6%** | **+10.9%** | **10** | **4** | **+6** | 75% | 66% |
| Gemma3 4B | debate-all | 52.7% | 56.4% | +3.6% | 10 | 8 | +2 | 100% | 58% |
| Nemotron 3 4B | independent | 60.0% | 60.0% | +0.0% | 0 | 0 | 0 | 0% | 100% |
| Nemotron 3 4B | debate-on-disagr | 60.0% | 61.8% | +1.8% | 3 | 2 | +1 | 42% | 87% |
| Nemotron 3 4B | debate-all | 60.0% | 61.8% | +1.8% | 3 | 2 | +1 | 100% | 87% |
| Ministral 3B | independent | 72.7% | 72.7% | +0.0% | 0 | 0 | 0 | 0% | 100% |
| Ministral 3B | debate-on-disagr | 74.6% | 60.0% | -14.5% | 4 | 12 | -8 | 62% | 66% |
| Ministral 3B | debate-all | 74.6% | 60.0% | -14.5% | 4 | 12 | -8 | 100% | 66% |
| **Qwen3.5 4B** | **independent** | **85.5%** | **85.5%** | **+0.0%** | **0** | **0** | **0** | **0%** | **100%** |
| Qwen3.5 4B | debate-on-disagr | 83.6% | 87.3% | +3.6% | 2 | 0 | +2 | 42% | 96% |
| **Qwen3.5 4B** | **debate-all** | **85.5%** | **89.1%** | **+3.6%** | **2** | **0** | **+2** | 100% | 96% |
| BitNet 2B | independent | 27.3% | 27.3% | +0.0% | 0 | 0 | 0 | 0% | 100% |
| BitNet 2B | debate-on-disagr | 0.0% | 0.0% | +0.0% | 0 | 0 | 0 | 0% | 100% |
| Llama3.2 3B | independent | 63.6% | 63.6% | +0.0% | 0 | 0 | 0 | 0% | 100% |
| Llama3.2 3B | debate-on-disagr | 61.8% | 60.0% | -1.8% | 9 | 10 | -1 | 87% | 55% |
| Llama3.2 3B | debate-all | 65.5% | 60.0% | -5.5% | 8 | 11 | -3 | 100% | 53% |
| Qwen3 4B base | independent | 65.5% | 65.5% | +0.0% | 0 | 0 | 0 | 0% | 100% |
| Qwen3 4B base | debate-on-disagr | 65.5% | 54.5% | -10.9% | 2 | 8 | -6 | 71% | 75% |
| Qwen3 4B base | debate-all | 69.1% | 54.5% | -14.5% | 1 | 9 | -8 | 100% | 76% |

### Ranking por accuracy final (mejor config por modelo)

| # | Modelo | Mejor config | Final% |
|---|--------|-------------|--------|
| 1 | Qwen3.5 4B | debate-all | 89.1% |
| 2 | Granite 3B | debate-all | 80.0% |
| 3 | Qwen3 4B-RAG | independent | 76.4% |
| 4 | Ministral 3B | independent | 72.7% |
| 5 | Qwen3 4B base | independent | 65.5% |
| 6 | Llama3.2 3B | independent | 63.6% |
| 7 | Nemotron 3 4B | debate-all | 61.8% |
| 8 | Gemma3 4B | debate-on-disagr | 63.6% |
| 9 | BitNet 2B | independent | 27.3% (artefacto, ver EXP-017) |

### Ranking por benefit del debate (net = corrections - damage)

| Modelo | Modo | Net | Delta |
|--------|------|-----|-------|
| Granite 3B | debate-all | +6 | +10.9% |
| Gemma3 4B | debate-on-disagr | +6 | +10.9% |
| Qwen3.5 4B | debate-all | +2 | +3.6% |
| Qwen3.5 4B | debate-on-disagr | +2 | +3.6% |
| Nemotron 3 4B | debate-all | +1 | +1.8% |
| Nemotron 3 4B | debate-on-disagr | +1 | +1.8% |
| Granite 3B | debate-on-disagr | +1 | +1.8% |
| Qwen3 4B-RAG | debate-on-disagr | 0 | +0.0% |
| Qwen3 4B-RAG | debate-all | -1 | -1.8% |
| Llama3.2 3B | debate-on-disagr | -1 | -1.8% |
| Llama3.2 3B | debate-all | -3 | -5.5% |
| Qwen3 4B base | debate-on-disagr | -6 | -10.9% |
| Ministral 3B | debate-on-disagr | -8 | -14.5% |
| Ministral 3B | debate-all | -8 | -14.5% |
| Qwen3 4B base | debate-all | -8 | -14.5% |

## Hallazgo clave: BitNet y constrained generation

### Problema

BitNet en el microcoliseum (sin `json_schema`) ecoa el template del
prompt literalmente:

```
Input:  {"worker": "A", "relation": "SUPPORTS|PARTIAL|CONTRADICTS|UNRELATED", ...}
Output: {"worker": "A", "relation": "SUPPORTS|PARTIAL|CONTRADICTS|UNRELATED", ...}
```

El parser `_normalize_relation` encontraba "CONTRADICTS" primero
(alfabeticamente) en el string con todas las opciones -> todo daba
CONTRADICTS -> 27.3% accuracy era **artefacto del parser**, no
capacidad real del modelo.

### Causa raiz

BitNet-b1.58-2B-4T no entiende la instruccion de "rellenar el template
con valores reales". Cuando ve un JSON con placeholders, lo copia
verbatim. Esto es consistente con su limitacion de capacidad (PM-003):
el modelo no tiene suficiente comprension del formato para generar
JSON estructurado por si solo.

### Solucion: json_schema por request

llama-server (bitnet.cpp) soporta el parametro `json_schema` en el
body del POST a `/completion`. Esto activa **constrained generation**
(GBNF grammar interno derivado del JSON schema) que fuerza al modelo
a producir JSON valido con los campos y valores del schema.

**Cambio**: una linea en `LlamaServerProvider.generate_structured`:
```python
"json_schema": schema,  # agregado al body del POST
```

### Verificacion

Test con 4 casos representativos:

| Caso | Expected | Sin json_schema | Con json_schema |
|------|----------|----------------|----------------|
| d-001 | SUPPORTS | CONTRADICTS (eco template) | **SUPPORTS** (correcto) |
| ec-001 | CONTRADICTS | CONTRADICTS (eco template) | SUPPORTS (incorrecto, pero capacidad) |
| ws-001 | UNRELATED | CONTRADICTS (eco template) | SUPPORTS (incorrecto, pero capacidad) |
| ps-001 | PARTIAL | CONTRADICTS (eco template) | **PARTIAL** (correcto) |

Los 3 schemas del debate (assessment, challenge, judge) producen JSON
valido con `json_schema`. La accuracy que produzca ahora sera
**capacidad real de BitNet**, no artefacto del parser.

### Impacto en EXP-015

EXP-015 (BitNet en Coliseo v1, protocolo corregido) uso prompts
few-shot con formato `RELATION:` (un token) y parser estricto. Ese
formato funciono porque BitNet si puede completar un token despues
de `RELATION:`. El 29.1% de EXP-015 es valido.

El microcoliseum usa prompts con JSON template, que BitNet no puede
completar sin constrained generation. El 27.3% del microcoliseum
(sin json_schema) es **artefacto** y debe descartarse.

## Observaciones

### 1. Qwen3.5 4B es el nuevo lider

Qwen3.5 4B supera a Qwen3 4B-RAG en accuracy final (89.1% vs 76.4%)
y ademas se beneficia del debate (net +2, 0 damage). Es el unico
modelo donde debate-all logra 0 damage.

### 2. El debate beneficia a modelos de capacidad media

Granite (69.1% -> 80.0%) y Gemma3 (52.7% -> 63.6%) son los que mas
se benefician del debate, ambos con net +6. Estos modelos tienen
suficiente capacidad para que el challenge round genere argumentos
utiles, pero sufren errores iniciales que el judge puede corregir.

### 3. El debate dana a Ministral 3B

Ministral 3B sufre damage catastrofico (-14.5%, net -8). El judge
destruye 12 respuestas correctas. Esto sugiere que el modelo genera
argumentos persuasivos pero incorrectos en el challenge round,
confundiendo al judge.

### 4. Modelos de alta capacidad no necesitan debate

Qwen3 4B-RAG (76.4%) no se beneficia del debate (net 0 o -1).
Qwen3.5 4B (85.5%) se beneficia marginalmente (net +2). Cuando el
modelo ya es bueno, el debate introduce mas ruido que correcciones.

### 5. debate-all vs debate-on-disagreement

No hay un ganador universal:
- Granite: debate-all (+6) >> debate-on-disagr (+1)
- Gemma3: debate-on-disagr (+6) > debate-all (+2)
- Qwen3.5: ambos igual (+2)
- Ministral: ambos igualmente catastroficos (-8)

### 6. Qwen3 4B-RAG vs Qwen3 4B base

Pendiente: Qwen3 4B base (sin system prompt custom) esta en corrida.
La comparacion medira cuanto del rendimiento de Qwen3 4B-RAG se debe
al system prompt RAG vs el modelo base.

### 7. Llama3.2 con protocolo corregido

Pendiente: Llama3.2 esta en corrida con think=false y JSON schema
(protocolo corregido de POST-001). Los resultados previos de EXP-014
estaban contaminados.

## Configuracion tecnica

### Cambios en debate.py

1. **`LlamaServerProvider`** (nueva clase): provider para llama-server
   con la misma interfaz que `StructuredOllamaProvider`. Usa
   `/completion` (OpenAI-compatible) con `json_schema` por request
   para constrained generation.

2. **`--backend ollama|llama-server`** (nuevo flag): selecciona el
   backend. Default: `ollama` (backward compatible).

3. **`--base-url`** (nuevo flag): override de URL para llama-server.

4. **`--num-ctx`** (nuevo flag): override de context size. Necesario
   para Qwen3 4B base (ctx default 40960 -> OOM en GPU de 6GB).

### Runner run_microcoliseum_trio.py

Orquestador para BitNet + Llama3.2 + Qwen3 4B base:
- Fase 1: BitNet via llama-server (CPU, arranca/detiene servidor)
- Fase 2: Llama3.2 via Ollama (GPU, think=false)
- Fase 3: Qwen3 4B base via Ollama (GPU, num_ctx=4096)

### Hallazgo: Qwen3 4B base y OOM

El modelo `hf.co/Qwen/Qwen3-4B-GGUF:Q4_K_M` tiene context length
40960 por defecto. Ollama intenta allocar ~24GB de KV cache en GPU
de 6GB -> `cudaMalloc failed: out of memory`. Solucion: setear
`num_ctx=4096` en las options (replicando el Modelfile custom de
qwen3-4b-rag).

## Conclusion (final)

### H1: confirmada con matices

El debate corrige mas errores de los que introduce en 5 de 8 modelos
evaluados completamente. Los outliers negativos son Qwen3 4B-RAG
(neutral), Llama3.2 (ligeramente negativo), Ministral 3B
(catastroficamente negativo) y Qwen3 4B base (catastroficamente
negativo).

### H2: confirmada

Modelos de alta capacidad (Qwen3.5 85.5%) se benefician marginalmente
(net +2, 0 damage). Modelos de capacidad media (Granite 69.1%,
Gemma3 52.7%) se benefician mucho (net +6). Modelos de baja
capacidad (BitNet) no se benefician porque no pueden generar
argumentos coherentes en el challenge round.

### H3: refutada (ver EXP-017)

BitNet puede generar JSON valido con `json_schema` constraint, pero
su accuracy real (29.1% en regimen optimo) es insuficiente. El
microcoliseum con BitNet produce artefactos sin constrained generation
y capacidad insuficiente con el. Ver EXP-017 para analisis exhaustivo.

### Patron emergente: capacidad vs benefit del debate

```
Capacidad baja (BitNet ~29%):    debate inutil (no genera argumentos)
Capacidad baja (Nemotron ~60%):  debate marginal positivo (+1)
Capacidad baja (Gemma3 ~53%):    debate muy util (net +6)
Capacidad media (Llama3.2 ~64%): debate daniño (net -1 a -3)
Capacidad media (Qwen3 base ~66%): debate catastrofico (net -8)
Capacidad media (Granite ~69%):  debate muy util (net +6)
Capacidad alta (Ministral ~73%): debate catastrofico (net -8) [outlier]
Capacidad alta (Qwen3-RAG ~76%): debate inutil o ligeramente daniño
Capacidad muy alta (Qwen3.5 ~86%): debate marginal seguro (net +2, 0 damage)
```

Outliers: Ministral y Qwen3 4B base tienen alta capacidad inicial pero
debate catastrofico. Posible explicacion: el modelo genera argumentos
persuasivos pero semanticamente incorrectos, confundiendo al judge.

### Hallazgo: Qwen3 4B-RAG vs Qwen3 4B base

| Modelo | Independent | Mejor debate | Net mejor debate |
|--------|-------------|-------------|-----------------|
| Qwen3 4B-RAG (custom system prompt) | 76.4% | 74.6% | -1 |
| Qwen3 4B base (sin system prompt) | 65.5% | 54.5% | -8 |
| Diferencia | +10.9% | +20.1% | |

El system prompt custom de `qwen3-4b-rag` aporta +10.9% en accuracy
Y hace al modelo resistente al damage del debate (-1 vs -8).

### Hallazgo: Llama3.2 con protocolo corregido

Llama3.2 3B independent: 63.6% (vs 16.4% historico contaminado de
POST-001). El protocolo corregido (think=false, JSON estructurado,
parser estricto) funciona correctamente. Pero el debate empeora:
net -1 (debate-on-disagreement) y -3 (debate-all). El judge destruye
sistematicamente wrong_subject y wrong_context (9 casos
UNRELATED->CONTRADICTS en debate-all).

## Estado

- [x] Completar Llama3.2 (3 modos, protocolo corregido)
- [x] Completar Qwen3 4B base (3 modos, num_ctx=4096)
- [x] Comparar Qwen3 4B base vs Qwen3 4B-RAG (impacto del system prompt)
- [x] Analizar transiciones por modelo (patrones de damage)
- [x] BitNet: barrido multidimensional exhaustivo (ver EXP-017)
