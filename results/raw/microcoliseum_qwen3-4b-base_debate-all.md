# Micro-Coliseum - Deliberative Semantic Assessment

## Configuration

| Parameter | Value |
|---|---|
| Model | `hf.co/Qwen/Qwen3-4B-GGUF:Q4_K_M` |
| Mode | `debate-all` |
| Workers | 4 (A=entailment, B=skeptical, C=contradiction, D=context) |
| Cases | 55 |
| GPU | True |
| num_predict | 64 |
| temperature | 0.0 |
| Benchmark | semantic_assessment_benchmark_v2.json |
| Timestamp | 2026-08-17T02:41:39 |

## Accuracy

| Metric | Initial | Deliberative | Delta |
|---|---:|---:|---:|
| Accuracy | 69.1% | 54.5% | -14.5% |
| Correct | 38/55 | 30/55 | -8 |

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
| Corrections (wrong->right) | 1 |
| Correction rate | 5.9% |
| Damage (right->wrong) | 9 |
| Damage rate | 23.7% |
| Net effect | -8 |
| Stability rate | 76.4% |

## Debate Statistics

| Metric | Value |
|---|---:|
| Debates triggered | 55/55 |
| Debate trigger rate | 100.0% |
| Workers changed opinion | 47 |
| Revision rate | 21.4% |

## Accuracy by Category

| Category | N | Initial | Deliberative | Delta |
|---|---:|---:|---:|---:|
| adversarial | 6 | 66.7% | 16.7% | -50.0% |
| direct_evidence | 6 | 100.0% | 100.0% | +0.0% |
| explicit_contradiction | 5 | 100.0% | 100.0% | +0.0% |
| implicit_contradiction | 5 | 100.0% | 100.0% | +0.0% |
| negation | 6 | 83.3% | 100.0% | +16.7% |
| over_specificity | 5 | 40.0% | 20.0% | -20.0% |
| paraphrase | 6 | 100.0% | 100.0% | +0.0% |
| partial_support | 6 | 33.3% | 0.0% | -33.3% |
| wrong_context | 5 | 0.0% | 0.0% | +0.0% |
| wrong_subject | 5 | 60.0% | 0.0% | -60.0% |

## Accuracy by Agreement Level

| Agreement | N | Initial | Deliberative | Delta |
|---|---:|---:|---:|---:|
| split | 39 | 59.0% | 38.5% | -20.5% |
| unanimous | 16 | 93.8% | 93.8% | +0.0% |

## Initial -> Final Transition Matrix

```

           CONT  PART  SUPP  UNRE
         ------------------------
 CONTRAD    24     1     3     0
 PARTIAL     0     1     6     0
 SUPPORT     0     0    17     0
 UNRELAT     2     1     0     0
```

## Case-level Results

| ID | Category | Expected | Initial | Final | Init OK | Final OK | Debate | Changed |
|---|---|---|---|---|---|---|---|---|
| d-001 | direct_evidence | SUPPORTS | SUPPORTS | SUPPORTS | Y | Y | Y | B |
| d-002 | direct_evidence | SUPPORTS | SUPPORTS | SUPPORTS | Y | Y | Y | - |
| d-003 | direct_evidence | SUPPORTS | SUPPORTS | SUPPORTS | Y | Y | Y | - |
| d-004 | direct_evidence | SUPPORTS | SUPPORTS | SUPPORTS | Y | Y | Y | - |
| d-005 | direct_evidence | SUPPORTS | SUPPORTS | SUPPORTS | Y | Y | Y | C |
| d-006 | direct_evidence | SUPPORTS | SUPPORTS | SUPPORTS | Y | Y | Y | B |
| pp-001 | paraphrase | SUPPORTS | SUPPORTS | SUPPORTS | Y | Y | Y | - |
| pp-002 | paraphrase | SUPPORTS | SUPPORTS | SUPPORTS | Y | Y | Y | B |
| pp-003 | paraphrase | SUPPORTS | SUPPORTS | SUPPORTS | Y | Y | Y | - |
| pp-004 | paraphrase | SUPPORTS | SUPPORTS | SUPPORTS | Y | Y | Y | B |
| pp-005 | paraphrase | SUPPORTS | SUPPORTS | SUPPORTS | Y | Y | Y | - |
| pp-006 | paraphrase | SUPPORTS | SUPPORTS | SUPPORTS | Y | Y | Y | A,B |
| ps-001 | partial_support | PARTIAL | PARTIAL | SUPPORTS | Y | N | Y | B |
| ps-002 | partial_support | PARTIAL | SUPPORTS | SUPPORTS | N | N | Y | A,B |
| ps-003 | partial_support | PARTIAL | SUPPORTS | SUPPORTS | N | N | Y | - |
| ps-004 | partial_support | PARTIAL | PARTIAL | SUPPORTS | Y | N | Y | A |
| ps-005 | partial_support | PARTIAL | CONTRADICTS | SUPPORTS | N | N | Y | A,B |
| ps-006 | partial_support | PARTIAL | CONTRADICTS | SUPPORTS | N | N | Y | A,B |
| ec-001 | explicit_contradiction | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | Y | - |
| ec-002 | explicit_contradiction | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | Y | - |
| ec-003 | explicit_contradiction | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | Y | - |
| ec-004 | explicit_contradiction | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | Y | - |
| ec-005 | explicit_contradiction | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | Y | - |
| ic-001 | implicit_contradiction | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | Y | - |
| ic-002 | implicit_contradiction | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | Y | - |
| ic-003 | implicit_contradiction | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | Y | A |
| ic-004 | implicit_contradiction | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | Y | - |
| ic-005 | implicit_contradiction | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | Y | - |
| os-001 | over_specificity | PARTIAL | CONTRADICTS | SUPPORTS | N | N | Y | - |
| os-002 | over_specificity | PARTIAL | SUPPORTS | SUPPORTS | N | N | Y | B |
| os-003 | over_specificity | PARTIAL | PARTIAL | SUPPORTS | Y | N | Y | A |
| os-004 | over_specificity | PARTIAL | SUPPORTS | SUPPORTS | N | N | Y | - |
| os-005 | over_specificity | PARTIAL | PARTIAL | PARTIAL | Y | Y | Y | - |
| n-001 | negation | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | Y | D |
| n-002 | negation | PARTIAL | CONTRADICTS | PARTIAL | N | Y | Y | A |
| n-003 | negation | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | Y | - |
| n-004 | negation | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | Y | - |
| n-005 | negation | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | Y | - |
| n-006 | negation | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | Y | D |
| ws-001 | wrong_subject | UNRELATED | CONTRADICTS | CONTRADICTS | N | N | Y | A |
| ws-002 | wrong_subject | UNRELATED | UNRELATED | CONTRADICTS | Y | N | Y | A,B,D |
| ws-003 | wrong_subject | UNRELATED | UNRELATED | PARTIAL | Y | N | Y | A,B,D |
| ws-004 | wrong_subject | UNRELATED | CONTRADICTS | CONTRADICTS | N | N | Y | A,D |
| ws-005 | wrong_subject | UNRELATED | UNRELATED | CONTRADICTS | Y | N | Y | A,B,D |
| wc-001 | wrong_context | UNRELATED | CONTRADICTS | CONTRADICTS | N | N | Y | A |
| wc-002 | wrong_context | UNRELATED | CONTRADICTS | CONTRADICTS | N | N | Y | A |
| wc-003 | wrong_context | UNRELATED | CONTRADICTS | CONTRADICTS | N | N | Y | A,B,D |
| wc-004 | wrong_context | UNRELATED | CONTRADICTS | CONTRADICTS | N | N | Y | A,B |
| wc-005 | wrong_context | PARTIAL | CONTRADICTS | CONTRADICTS | N | N | Y | A |
| a-001 | adversarial | PARTIAL | PARTIAL | SUPPORTS | Y | N | Y | A |
| a-002 | adversarial | PARTIAL | PARTIAL | SUPPORTS | Y | N | Y | A |
| a-003 | adversarial | PARTIAL | SUPPORTS | SUPPORTS | N | N | Y | A |
| a-004 | adversarial | PARTIAL | CONTRADICTS | CONTRADICTS | N | N | Y | A |
| a-005 | adversarial | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | Y | - |
| a-006 | adversarial | PARTIAL | PARTIAL | SUPPORTS | Y | N | Y | A,B |

## Conclusion

Mode: `debate-all`. The deliberation produced a **net negative** effect: 1 corrections vs 9 damage (net -8). Accuracy changed from 69.1% to 54.5% (-14.5%). The debate introduced more errors than it corrected. H0 is supported.
