# Small Model Deliberation

Can small LLMs (3-4B params) produce reliable semantic assessments through
deliberative interaction, and does debate between workers correct errors
that parallel ensemble voting cannot?

## Research Question

The central question is not "which model is best" but rather:

> Does deliberative interaction between semantic experts produce a
> better signal than independent aggregation?

This repo contains the complete experimental chain from initial BitNet
evaluation through deliberative micro-coliseum, with all raw data,
scripts, and analysis.

## Structure

```
small-model-deliberation/
|
|-- README.md
|
|-- docs/
|   |-- 00-research-question.md       Research framing
|   |-- 01-initial-hypothesis.md      Model strategy (3B vs 8B)
|   |-- 02-bitnet-experiment.md       EXP-010: BitNet capacity
|   |-- 03-coliseum-v1.md             EXP-011/012: Granite vs BitNet, Coliseo v1
|   |-- 04-ensemble-analysis.md       EXP-013: CPU vs GPU
|   |-- 05-microcoliseum.md           EXP-014: Deliberative debate
|   |-- 06-debate.md                  Debate analysis (pending)
|   |-- 07-final-findings.md          Final findings (pending)
|   |-- adr-0031-llmsupport.md        ADR reference
|   |-- res-003-knowledge-consumer.md Research note
|   |-- res-004-llmsupport-observador.md  Research note
|   |-- res-007-model-strategy.md     Research note
|   |-- res-016-bitnet-vision.md      Research note
|
|-- benchmarks/
|   |-- semantic_assessment_v1.json   12 cases (original)
|   |-- semantic_assessment_v2.json   55 cases, 10 diagnostic categories
|
|-- experiments/
|   |-- EXP-001-bitnet/               BitNet ensemble (EXP-010)
|   |-- EXP-002-coliseum-v1/          Granite vs BitNet + Coliseo v1 CPU (EXP-011/012)
|   |-- EXP-003-gpu/                  Coliseo v1 GPU (EXP-013)
|   |-- EXP-004-debate/               Deliberative Micro-Coliseum (EXP-014)
|   |-- POST-001-protocol-contamination.md  Postmortem: protocol contamination
|   |-- PAT-001-ollama-multi-instance.md    Pattern: multi-port Ollama instances
|
|-- runners/
|   |-- ollama_instances.py           Multi-instance Ollama manager (PAT-001)
|   |-- ensemble.py                   Ensemble runner (coliseo)
|   |-- debate.py                     Deliberative micro-coliseum runner
|   |-- judge.py                      Judge logic (embedded in debate.py)
|   |-- run_coliseo_v1_gpu.py         Coliseo v1 GPU (--port aware)
|   |-- run_coliseo_v2_gpu.py         Coliseo v2 GPU (--port aware)
|   |-- run_coliseo_v1_llama32_cpu_controlled.py  Llama3.2 controlled re-run
|   |-- run_microcoliseum_all.py      Batch runner for microcoliseum (--port aware)
|   |-- ...                           Other pilot scripts
|
|-- results/
|   |-- raw/                          Raw JSON outputs from every run
|   |-- processed/                    Processed analysis (pending)
|
|-- analysis/                         Analysis scripts and notebooks
|
|-- paper/
|   |-- paper.md                      Paper draft (pending)
|   |-- figures/                      Generated figures (pending)
```

## Setup

### Prerequisites

1. **Python 3.10+**
2. **Ollama** installed and running (for 3B-4B models)
3. **bitnet.cpp** built (for BitNet experiments only) — set `BITNET_ROOT` env var:
   ```bash
   # Linux/macOS
   export BITNET_ROOT=/path/to/BitNet
   # Windows PowerShell
   $env:BITNET_ROOT = "C:\path\to\BitNet"
   ```
4. **hybrid_rag package** — this repo depends on the `hybrid_rag` package from
   the AgenticRAG project. Install it as an editable package:
   ```bash
   cd /path/to/AgenticRAG
   pip install -e .
   ```

### Install

```bash
pip install -r requirements.txt
```

### Ollama models

Pull the models used in the experiments:

```bash
ollama pull llama3.2:3b
ollama pull gemma3:4b-it-q4_K_M
ollama pull qwen3.5:4b-q4_K_M
ollama pull hf.co/Qwen/Qwen3-4B-GGUF:Q4_K_M
ollama pull ibm/granite4.1:3b-q4_K_M
ollama pull TechyShishy/ministral-3:3b-reasoning-2512-q4_K_M
ollama pull dhiltgen/nemotron-3-nano:4b
```

### Multi-instance Ollama (for parallel experiments)

See `experiments/PAT-001-ollama-multi-instance.md` for the multi-instance pattern.

## Experiment Chain

| EXP | Title | Key Finding |
|-----|-------|-------------|
| EXP-010 | BitNet ensemble | 2B ternary insufficient (33-50% accuracy). Ensemble worse than single. |
| EXP-011 | Granite vs BitNet | Granite 3B Q4: 91.7% single, 100% ensemble on v1 dataset. |
| EXP-012 | Coliseo v1 (CPU) | Qwen3 4B-RAG wins: 83.6% ensemble_2. ~~Llama 3.2 3B fails (16.4%).~~ **CONTAMINADO — ver POST-001.** |
| EXP-013 | Coliseo v1 (GPU) | GPU = speed only, no quality change. 2.9x speedup on ensemble_4. Llama3.2 results contaminated. |
| EXP-014 | Deliberative Micro-Coliseum | Granite debate-all: 12 corrections, 6 damage, net +6 (+10.9%). Qwen3 4B-RAG no benefit (net 0/-1). |
| EXP-015 | BitNet Coliseo v1 (protocolo corregido) | BitNet NO fue contaminado: 29.1% single en 55 casos confirma PM-003. Condena se sostiene. |
| EXP-016 | Micro-Coliseum extendido (9 modelos) | Qwen3.5 4B lider (89.1% debate-all, 0 damage). Granite +6, Gemma3 +6. Ministral y Qwen3 base debate catastrofico (-14.5%). Llama3.2 debate daniño (-5.5%). System prompt de Qwen3-RAG aporta +10.9% y resistencia al damage. |
| EXP-017 | BitNet barrido multidimensional | 7 configuraciones (3 regimenes x decoding x ensemble), ninguna supera 29.1%. Perfil cognitivo: 0% entailment/paraphrase, 80% explicit_contradiction. Veredicto: cuantizacion 1.58 bits destruye entailment. |
| EXP-018 | BitNet NLI reframing + logit ensemble | NLI reframing (TRUE/FALSE/CANNOT_TELL) rompe la SUPPORTS wall: 12/12 SUPPORTS correctos. Logit ensemble (logsumexp) alcanza 40.0% (+10.9% sobre EXP-017). 50% no alcanzable: NLI gana entailment pero pierde relevancia detection. |
| EXP-019 | BitNet relevance x entailment decomposition | Sistema de 2 etapas (relevance gate + entailment + decision layer). Relevance detection: 100% irrelevantes claros. Hybrid con EXP-018: 43.6% (+3.6%). Frontera: granularity assessment (PARTIAL) ausente, relevance sutil debil. |
| EXP-020 | BitNet granularity probe + atomic decomposition | Casos minimos controlados. Comportamiento consistente con keyword matching holistico, sin verificacion composicional confiable. Atomic FALSE accuracy: 0%. La frontera es verificacion de ausencia. Atomic decomposition no funciona. |
| EXP-021 | BitNet absence detection falsation probe | Experimento de falsacion final. 3 condiciones minimas. FALSE accuracy: 0/12 (0%). BitNet dice TRUE sistematicamente para claims no-soportados. Margen siempre positivo. Incapacidad operacional confirmada. Cierre de rama. |
| EXP-022 | BitNet microcoliseum especializado | 4 workers especializados + judge deterministico. 29.1%, +5.5% sobre ensemble. Contradicciones 20%→60%. Especializacion parcialmente artefacto del grammar (EXP-023). |
| EXP-023 | BitNet grammar sensitivity probe | **Grammar es variable de primer orden**: G1 estricto → TRUE x55, G2 permisivo → FALSE x53, inversion completa. Los 12/12 SUPPORTS de EXP-018 fueron artefacto del grammar estricto. |
| EXP-024 | BitNet semantic discrimination probe | **No hay señal semantica**: P(TRUE\|SUPPORTS) ≈ P(TRUE\|CONTRADICTS), delta = +0.0017. BitNet sale del pipeline semantico. PM-003 cerrado definitivamente. |
| POST-001 | Protocol contamination postmortem | Nemotron, Ministral, Qwen3.5, Llama3.2 had artificially low scores due to num_predict=10 + think mode + lenient parser. Llama3.2 re-run: 58.2% single (vs 16.4% historical). BitNet confirmed NOT contaminated. |
| PAT-001 | Ollama multi-instance pattern | Multi-port Ollama instances to avoid model thrashing in parallel experiments. |

## Key Results

### Coliseo v1 (EXP-012)

| Model | single | ensemble_2 | ensemble_4 |
|-------|--------|------------|------------|
| granite-3b-q4 | 61.8% | 67.3% | 76.4% |
| llama32-3b | ~~16.4%~~ → **58.2%** (POST-001) | ~~29.1%~~ → (en progreso) | ~~20.0%~~ → (en progreso) |
| **qwen3-4b-rag** | **78.2%** | **83.6%** | 81.8% |

> **POST-001:** Los resultados originales de Llama3.2 (16.4%/29.1%/20.0%)
> estaban contaminados por `num_predict=10` + parser leniento. La
> repeticion con protocolo corregido (`num_predict=64`, `think=false`,
> JSON estructurado, parser estricto) da **58.2% single**. La conclusion
> de "Llama3.2 es semanticamente insuficiente" ha sido retirada.

### Deliberative Micro-Coliseum (EXP-014 + EXP-016)

**Ranking por accuracy final (mejor config por modelo, 9 modelos):**

| # | Model | Best config | Init% | Final% | Net |
|---|-------|-------------|-------|--------|-----|
| 1 | Qwen3.5 4B | debate-all | 85.5% | **89.1%** | +2 |
| 2 | Granite 3B | debate-all | 69.1% | 80.0% | +6 |
| 3 | Qwen3 4B-RAG | independent | 76.4% | 76.4% | 0 |
| 4 | Ministral 3B | independent | 72.7% | 72.7% | 0 |
| 5 | Qwen3 4B base | independent | 65.5% | 65.5% | 0 |
| 6 | Llama3.2 3B | independent | 63.6% | 63.6% | 0 |
| 7 | Gemma3 4B | debate-on-disagr | 52.7% | 63.6% | +6 |
| 8 | Nemotron 3 4B | debate-all | 60.0% | 61.8% | +1 |
| 9 | BitNet 2B | independent | 29.1% | 29.1% | 0 |

**Key findings:**

1. **Qwen3.5 4B fue el mejor candidato evaluado** (89.1% debate, 2/55
   casos recuperados, 0 danos en esta corrida). Supera a Qwen3 4B-RAG
   por ~13 puntos. Limitaciones: un solo benchmark, N=55, un dominio,
   seed limitada (ver paper para discusion completa).
2. **El debate no es monotonicamente beneficioso**: beneficia a modelos
   de capacidad media (Granite +6, Gemma3 +6) pero dana a Ministral
   (-8) y Qwen3 4B base (-8). Es un componente sometido a evidencia,
   no una presuposicion de "mas reasoning = mejor".
3. **BitNet b1.58-2B-4T descartado**: 8 experimentos (EXP-017 a EXP-024)
   demostraron ausencia de señal semantica discriminativa (delta=0.0017).
   El comportamiento observable es lexical separability, no semantic
   relevance detection.
4. **Regla metodologica**: primero demostrar la señal, despues asignarle
   ownership, recien entonces construir la arquitectura alrededor de ella.
   El MoE Qwen3.5 + Qwen3 RAG es hipotesis pendiente de validacion fuera
   de muestra, no arquitectura demostrada.
5. **Llama3.2 con protocolo corregido**: 63.6% independent (vs 16.4%
   historico contaminado). El protocolo corregido funciona, pero el
   debate empeora (net -3 en debate-all).
6. **BitNet requiere GBNF constraint** en llama-server: sin
   constrained generation, ecoa el template del prompt. Ver EXP-017
   para barrido multidimensional exhaustivo (7 configuraciones,
   ninguna supera 29.1%).

### BitNet barrido multidimensional (EXP-017)

| Configuracion | Accuracy |
|--------------|----------|
| Zero-Shot GBNF (temp=0.0) | 25.4% |
| Zero-Shot GBNF (temp=0.2, top_k=20) | 21.8% |
| **Few-Shot GBNF (temp=0.0, rep=1.0) [optimo]** | **29.1%** |
| Few-Shot GBNF (temp=0.2, top_k=20) | 25.4% |
| Binary Cascading / One-vs-All | 27.3% |
| Ensemble_2 (voting) | 27.3% |
| Ensemble_4 (voting) | 27.3% |

**Perfil cognitivo BitNet vs Qwen3.5 (55 casos, 10 categorias):**

| Categoria | BitNet | Qwen3.5 |
|-----------|--------|---------|
| direct_evidence | 0.0% | 100.0% |
| paraphrase | 0.0% | 100.0% |
| adversarial | 0.0% | 83.3% |
| explicit_contradiction | 80.0% | 100.0% |
| **TOTAL** | **29.1%** | **89.1%** |

**Veredicto**: La cuantizacion 1.58 bits destruye la capacidad de
entailment. Ningun regimen de inferencia hace a BitNet viable.
PM-003 confirmado de forma multidimensional.

### BitNet NLI reframing + logit ensemble (EXP-018)

| Configuracion | Accuracy |
|--------------|----------|
| Techo single regime (EXP-017) | 29.1% (16/55) |
| NLI 4-way single (greedy) | 29.1% (16/55) |
| Logit ensemble (fs0 + nli4 + bin, logsumexp) | 38.2% (21/55) |
| **Logit ensemble 4-regimen calibrado** | **40.0% (22/55)** |

**Hallazgos clave:**

1. **NLI reframing rompe la SUPPORTS wall**: reformulando a
   TRUE/FALSE/CANNOT_TELL, los 3 regimenes NLI aciertan 12/12 SUPPORTS.
   El sesgo era de token, no de capacidad subyacente.
2. **Logprobs revelan informacion oculta**: en casos de contradiccion,
   el token `False` tiene mayor logprob que `True`, pero el GBNF fuerza
   `TRUE`. El modelo distingue, el grammar no.
3. **Logit ensemble captura complementariedad**: fs0 detecta
   contradicciones, NLI detecta soporte, bin detecta partial. Logsumexp
   preserva confianza; voting la destruia.
4. **50% no alcanzable**: NLI gana entailment pero pierde relevancia
   detection (wrong_subject 60%→0%, wrong_context 40%→0%). Trade-off
   fundamental para 2B ternario.

**Veredicto revisado**: BitNet **si tiene** capacidad de entailment
cuando se reformula la tarea. Lo que se observa debilitado es la
capacidad de usar etiquetas artificiales, distinguir relevancia de
soporte en casos sutiles, y emitir PARTIAL con confianza. La causa no
esta aislada (puede ser cuantizacion, capacidad, entrenamiento, framing,
o combinacion). PM-003 se sostiene (43.6% < 60% minimo) pero con matiz.
Ver EXP-018 y EXP-019 para detalles.

### BitNet relevance x entailment decomposition (EXP-019)

Sistema de 2 etapas con decision layer deterministico:

| Capacidad | Medicion | Estado |
|-----------|----------|--------|
| Entailment binario (TRUE/FALSE) | 12/12 SUPPORTS, 5/5 contradicts | Fuerte |
| Relevance detection (claro) | 100% wrong_subject + wrong_context | Fuerte |
| Relevance detection (sutil) | ws-003, wc-003 pasan el gate | Debil |
| Granularity assessment (PARTIAL) | 0/17 en todos los experimentos | Ausente |

**Techo hybrid (EXP-018 ensemble + relevance gate): 43.6% (24/55)**

La frontera de BitNet esta en granularity assessment (PARTIAL) y
relevance sutil, no en entailment ni en relevance detection claro.

### BitNet granularity probe (EXP-020)

Casos minimos controlados para aislar granularity assessment:

| Capacidad | Medicion | Estado |
|-----------|----------|--------|
| Entailment binario (TRUE/FALSE) | 12/12 SUPPORTS | Fuerte |
| Relevance detection (claro) | 100% irrelevantes | Fuerte |
| **Verificacion de ausencia** | **0% FALSE correctos** | **Ausente** |
| Granularity assessment (PARTIAL) | 0/17 en todos los experimentos | Ausente |

**El comportamiento observable es consistente con keyword matching
holistico basado en overlap semantico/lexical, sin evidencia de
verificacion composicional confiable.** BitNet dice TRUE para
cualquier proposicion que comparta keywords con el evidence, sin
verificar ausencia. La atomic decomposition no funciona porque el
aggregator no puede componer señales correctas partiendo de señales
incorrectas (todas TRUE).

**Conclusion de la cadena experimental (EXP-017 -> 021):**

La frontera definitiva (bajo el protocolo actual) es la verificacion
de ausencia. EXP-021 confirmo con falsacion: FALSE accuracy 0/12,
BitNet dice TRUE sistematicamente para claims no-soportados, margen
siempre positivo (medio +0.776). Incapacidad operacional confirmada.

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
 relevance        NO       sutil    negativa   composicional
    |               |      |        |               |
   [OK]           [NO]   [NO]      [FAIL]         [FAIL]
```

**EXP-024 definitivo:** No hay señal semantica en la distribucion de
probabilidades. P(TRUE|SUPPORTS) ≈ P(TRUE|CONTRADICTS) ≈ 0.40, delta
= +0.0017. BitNet no discrimina semanticamente. El "entailment" de
EXP-018 fue un artefacto del grammar. BitNet sale del pipeline
semantico.

**PM-003: REJECTED (definitivo).** Los experimentos EXP-017 a EXP-024
no encontraron señal semantica explotable en BitNet 1.58b, ni en el
greedy decoding, ni en la distribucion de probabilidades, ni bajo
ningun grammar. El modelo no tiene caso de uso en el pipeline
semantico.

**Decision:** no continuar optimizando BitNet como juez semantico
generalista. **Uso potencial:** componente especializado de bajo
costo para señales semanticas elementales, subordinado a contratos
y politicas deterministas.

### Consecuencia arquitectonica

La cadena experimental reforzo la separacion fundamental: los LLMs
producen señales semanticas, la capa deterministica (contratos,
politicas, invariantes) ejerce autoridad. BitNet puede generar "hay
evidencia relacionada" (senal) pero no decidir "Evidence Contract
satisfied = TRUE" (autoridad). La arquitectura no necesita un juez
omnisciente — necesita señales gobernables.

## Raw Data

All raw JSON outputs are in `results/raw/`. Each file contains per-case
results including:

- Worker assessments (relation, confidence, reason)
- Initial ensemble result
- Challenge responses (counterarguments, change decisions)
- Judge final decision
- Ground truth
- Latency per phase

This makes every experiment auditable: you can trace why a case went
from incorrect to correct or vice versa.

## Status

- EXP-010 through EXP-013: completed
- EXP-014: completed (Granite + Qwen3 4B-RAG, 6 runs)
- EXP-015: completed — BitNet confirmed semantically insufficient (29.1% single, 55 cases)
- EXP-016: completed — Microcoliseum extendido (9 modelos). Qwen3.5 4B lider (89.1% debate-all)
- EXP-017: completed — BitNet barrido multidimensional (7 configs, techo 29.1%)
- EXP-018: completed — BitNet NLI reframing + logit ensemble (techo 40.0%, SUPPORTS wall rota)
- EXP-019: completed — BitNet relevance x entailment decomposition (techo 43.6% hybrid, frontier: PARTIAL ausente)
- EXP-020: completed — BitNet granularity probe + atomic decomposition (frontera: verificacion de ausencia ausente, comportamiento consistente con keyword matching holistico)
- EXP-021: completed — BitNet absence detection falsation probe (FALSE accuracy 0/12, cierre de rama, PM-003 cerrado)
- EXP-022: completed — BitNet microcoliseum especializado (4 workers + judge deterministico, 29.1%, +5.5% sobre ensemble, contradicciones 20%→60%)
- EXP-023: completed — BitNet grammar sensitivity probe (**grammar es variable de primer orden**: G1 estricto → TRUE x55, G2 permisivo → FALSE x53, inversion completa)
- EXP-024: completed — BitNet semantic discrimination probe (**no hay señal semantica**: P(TRUE|SUPPORTS) ≈ P(TRUE|CONTRADICTS), delta = +0.0017, BitNet sale del pipeline semantico)
- POST-001: completed — protocol contamination identified and fixed for Llama3.2; BitNet confirmed NOT contaminated
- PAT-001: active — multi-instance Ollama pattern for parallel experiments
- Coliseo v2 (new models: Gemma3, Nemotron, Ministral, Qwen3.5): in progress
- Llama3.2 controlled re-run (Coliseo v1, CPU, protocol fixed): completed (58.2% single, 63.6% ensemble_4)
- Re-evaluation of Nemotron, Ministral, Qwen3.5 with fixed protocol: pending
- Re-evaluation of Llama3.2 in Micro-Coliseum: in progress (EXP-016)
- BitNet re-run in Micro-Coliseum with json_schema fix: pending

## Origin

Extracted from the AgenticRAG project (hybrid-rag architecture).
The SemanticAssessment contract and SemanticEnsemble infrastructure
were originally developed there. This repo isolates the experimental
chain for reproducibility and future research.
