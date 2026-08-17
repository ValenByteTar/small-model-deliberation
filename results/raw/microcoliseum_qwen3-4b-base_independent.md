# Micro-Coliseum - Deliberative Semantic Assessment

## Configuration

| Parameter | Value |
|---|---|
| Model | `hf.co/Qwen/Qwen3-4B-GGUF:Q4_K_M` |
| Mode | `independent` |
| Workers | 4 (A=entailment, B=skeptical, C=contradiction, D=context) |
| Cases | 55 |
| GPU | True |
| num_predict | 64 |
| temperature | 0.0 |
| Benchmark | semantic_assessment_benchmark_v2.json |
| Timestamp | 2026-08-17T02:03:34 |

## Accuracy

| Metric | Initial | Deliberative | Delta |
|---|---:|---:|---:|
| Accuracy | 65.5% | 65.5% | +0.0% |
| Correct | 36/55 | 36/55 | +0 |

## Corrections vs Damage

```

                    FINAL
                     ^
        corrections  |  damage
                     |
INITIAL -------------+-------------
                     |
                 unchanged
```

| Metric | Value |
|---|---:|
| Corrections (wrong->right) | 0 |
| Correction rate | 0.0% |
| Damage (right->wrong) | 0 |
| Damage rate | 0.0% |
| Net effect | +0 |
| Stability rate | 100.0% |

## Debate Statistics

| Metric | Value |
|---|---:|
| Debates triggered | 0/55 |
| Debate trigger rate | 0.0% |
| Workers changed opinion | 0 |
| Revision rate | 0.0% |

## Accuracy by Category

| Category | N | Initial | Deliberative | Delta |
|---|---:|---:|---:|---:|
| adversarial | 6 | 50.0% | 50.0% | +0.0% |
| direct_evidence | 6 | 100.0% | 100.0% | +0.0% |
| explicit_contradiction | 5 | 100.0% | 100.0% | +0.0% |
| implicit_contradiction | 5 | 100.0% | 100.0% | +0.0% |
| negation | 6 | 83.3% | 83.3% | +0.0% |
| over_specificity | 5 | 60.0% | 60.0% | +0.0% |
| paraphrase | 6 | 83.3% | 83.3% | +0.0% |
| partial_support | 6 | 16.7% | 16.7% | +0.0% |
| wrong_context | 5 | 0.0% | 0.0% | +0.0% |
| wrong_subject | 5 | 60.0% | 60.0% | +0.0% |

## Accuracy by Agreement Level

| Agreement | N | Initial | Deliberative | Delta |
|---|---:|---:|---:|---:|
| split | 39 | 53.8% | 53.8% | +0.0% |
| unanimous | 16 | 93.8% | 93.8% | +0.0% |

## Initial -> Final Transition Matrix

```

           CONT  PART  SUPP  UNRE
         ------------------------
 CONTRAD    28     0     0     0
 PARTIAL     0     7     0     0
 SUPPORT     0     0    17     0
 UNRELAT     0     0     0     3
```

## Case-level Results

| ID | Category | Expected | Initial | Final | Init OK | Final OK | Debate | Changed |
|---|---|---|---|---|---|---|---|---|
| d-001 | direct_evidence | SUPPORTS | SUPPORTS | SUPPORTS | Y | Y | N | - |
| d-002 | direct_evidence | SUPPORTS | SUPPORTS | SUPPORTS | Y | Y | N | - |
| d-003 | direct_evidence | SUPPORTS | SUPPORTS | SUPPORTS | Y | Y | N | - |
| d-004 | direct_evidence | SUPPORTS | SUPPORTS | SUPPORTS | Y | Y | N | - |
| d-005 | direct_evidence | SUPPORTS | SUPPORTS | SUPPORTS | Y | Y | N | - |
| d-006 | direct_evidence | SUPPORTS | SUPPORTS | SUPPORTS | Y | Y | N | - |
| pp-001 | paraphrase | SUPPORTS | SUPPORTS | SUPPORTS | Y | Y | N | - |
| pp-002 | paraphrase | SUPPORTS | SUPPORTS | SUPPORTS | Y | Y | N | - |
| pp-003 | paraphrase | SUPPORTS | SUPPORTS | SUPPORTS | Y | Y | N | - |
| pp-004 | paraphrase | SUPPORTS | SUPPORTS | SUPPORTS | Y | Y | N | - |
| pp-005 | paraphrase | SUPPORTS | SUPPORTS | SUPPORTS | Y | Y | N | - |
| pp-006 | paraphrase | SUPPORTS | PARTIAL | PARTIAL | N | N | N | - |
| ps-001 | partial_support | PARTIAL | CONTRADICTS | CONTRADICTS | N | N | N | - |
| ps-002 | partial_support | PARTIAL | SUPPORTS | SUPPORTS | N | N | N | - |
| ps-003 | partial_support | PARTIAL | SUPPORTS | SUPPORTS | N | N | N | - |
| ps-004 | partial_support | PARTIAL | PARTIAL | PARTIAL | Y | Y | N | - |
| ps-005 | partial_support | PARTIAL | SUPPORTS | SUPPORTS | N | N | N | - |
| ps-006 | partial_support | PARTIAL | CONTRADICTS | CONTRADICTS | N | N | N | - |
| ec-001 | explicit_contradiction | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | N | - |
| ec-002 | explicit_contradiction | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | N | - |
| ec-003 | explicit_contradiction | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | N | - |
| ec-004 | explicit_contradiction | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | N | - |
| ec-005 | explicit_contradiction | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | N | - |
| ic-001 | implicit_contradiction | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | N | - |
| ic-002 | implicit_contradiction | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | N | - |
| ic-003 | implicit_contradiction | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | N | - |
| ic-004 | implicit_contradiction | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | N | - |
| ic-005 | implicit_contradiction | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | N | - |
| os-001 | over_specificity | PARTIAL | PARTIAL | PARTIAL | Y | Y | N | - |
| os-002 | over_specificity | PARTIAL | SUPPORTS | SUPPORTS | N | N | N | - |
| os-003 | over_specificity | PARTIAL | PARTIAL | PARTIAL | Y | Y | N | - |
| os-004 | over_specificity | PARTIAL | SUPPORTS | SUPPORTS | N | N | N | - |
| os-005 | over_specificity | PARTIAL | PARTIAL | PARTIAL | Y | Y | N | - |
| n-001 | negation | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | N | - |
| n-002 | negation | PARTIAL | CONTRADICTS | CONTRADICTS | N | N | N | - |
| n-003 | negation | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | N | - |
| n-004 | negation | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | N | - |
| n-005 | negation | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | N | - |
| n-006 | negation | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | N | - |
| ws-001 | wrong_subject | UNRELATED | CONTRADICTS | CONTRADICTS | N | N | N | - |
| ws-002 | wrong_subject | UNRELATED | UNRELATED | UNRELATED | Y | Y | N | - |
| ws-003 | wrong_subject | UNRELATED | UNRELATED | UNRELATED | Y | Y | N | - |
| ws-004 | wrong_subject | UNRELATED | CONTRADICTS | CONTRADICTS | N | N | N | - |
| ws-005 | wrong_subject | UNRELATED | UNRELATED | UNRELATED | Y | Y | N | - |
| wc-001 | wrong_context | UNRELATED | CONTRADICTS | CONTRADICTS | N | N | N | - |
| wc-002 | wrong_context | UNRELATED | CONTRADICTS | CONTRADICTS | N | N | N | - |
| wc-003 | wrong_context | UNRELATED | CONTRADICTS | CONTRADICTS | N | N | N | - |
| wc-004 | wrong_context | UNRELATED | CONTRADICTS | CONTRADICTS | N | N | N | - |
| wc-005 | wrong_context | PARTIAL | CONTRADICTS | CONTRADICTS | N | N | N | - |
| a-001 | adversarial | PARTIAL | PARTIAL | PARTIAL | Y | Y | N | - |
| a-002 | adversarial | PARTIAL | PARTIAL | PARTIAL | Y | Y | N | - |
| a-003 | adversarial | PARTIAL | SUPPORTS | SUPPORTS | N | N | N | - |
| a-004 | adversarial | PARTIAL | CONTRADICTS | CONTRADICTS | N | N | N | - |
| a-005 | adversarial | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | N | - |
| a-006 | adversarial | PARTIAL | CONTRADICTS | CONTRADICTS | N | N | N | - |

## Conclusion

Mode: `independent` (baseline, no debate). This run establishes the initial ensemble accuracy without deliberation.
