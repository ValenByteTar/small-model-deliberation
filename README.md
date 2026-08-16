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
|
|-- runners/
|   |-- ensemble.py                   Ensemble runner (coliseo)
|   |-- debate.py                     Deliberative micro-coliseum runner
|   |-- judge.py                      Judge logic (embedded in debate.py)
|   |-- run_coliseo_v1_gpu.py         Coliseo v1 GPU
|   |-- run_coliseo_v2_gpu.py         Coliseo v2 GPU (new models)
|   |-- run_microcoliseum_all.py      Batch runner for microcoliseum
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

## Experiment Chain

| EXP | Title | Key Finding |
|-----|-------|-------------|
| EXP-010 | BitNet ensemble | 2B ternary insufficient (33-50% accuracy). Ensemble worse than single. |
| EXP-011 | Granite vs BitNet | Granite 3B Q4: 91.7% single, 100% ensemble on v1 dataset. |
| EXP-012 | Coliseo v1 (CPU) | Qwen3 4B-RAG wins: 83.6% ensemble_2. Llama 3.2 3B fails (16.4%). |
| EXP-013 | Coliseo v1 (GPU) | GPU = speed only, no quality change. 2.9x speedup on ensemble_4. |
| EXP-014 | Deliberative Micro-Coliseum | Granite debate-all: 12 corrections, 6 damage, net +6 (+10.9%). |

## Key Results

### Coliseo v1 (EXP-012)

| Model | single | ensemble_2 | ensemble_4 |
|-------|--------|------------|------------|
| granite-3b-q4 | 61.8% | 67.3% | 76.4% |
| llama32-3b | 16.4% | 29.1% | 20.0% |
| **qwen3-4b-rag** | **78.2%** | **83.6%** | 81.8% |

### Deliberative Micro-Coliseum (EXP-014, Granite)

| Mode | Initial | Final | Delta | Corrections | Damage | Net |
|------|---------|-------|-------|-------------|--------|-----|
| independent | 69.1% | 69.1% | 0.0% | 0 | 0 | 0 |
| debate-on-disagreement | 67.3% | 69.1% | +1.8% | 12 | 11 | +1 |
| **debate-all** | **69.1%** | **80.0%** | **+10.9%** | **12** | **6** | **+6** |

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
- EXP-014: in-progress (Qwen3 and Llama 3.2 debate runs pending)
- Coliseo v2 (new models: Gemma3, Nemotron, Ministral, Qwen3.5): pending

## Origin

Extracted from the AgenticRAG project (hybrid-rag architecture).
The SemanticAssessment contract and SemanticEnsemble infrastructure
were originally developed there. This repo isolates the experimental
chain for reproducibility and future research.
