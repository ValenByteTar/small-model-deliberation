# Micro-Coliseum - Deliberative Semantic Assessment

## Configuration

| Parameter | Value |
|---|---|
| Model | `gemma3:4b-it-q4_K_M` |
| Mode | `debate-on-disagreement` |
| Workers | 4 (A=entailment, B=skeptical, C=contradiction, D=context) |
| Cases | 55 |
| GPU | True |
| num_predict | 60 |
| temperature | 0.0 |
| Benchmark | semantic_assessment_benchmark_v2.json |
| Timestamp | 2026-08-16T18:40:05 |

## Accuracy

| Metric | Initial | Deliberative | Delta |
|---|---:|---:|---:|
| Accuracy | 52.7% | 63.6% | +10.9% |
| Correct | 29/55 | 35/55 | +6 |

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
| Corrections (wrong->right) | 10 |
| Correction rate | 38.5% |
| Damage (right->wrong) | 4 |
| Damage rate | 13.8% |
| Net effect | +6 |
| Stability rate | 65.5% |

## Debate Statistics

| Metric | Value |
|---|---:|
| Debates triggered | 41/55 |
| Debate trigger rate | 74.6% |
| Workers changed opinion | 94 |
| Revision rate | 57.3% |

## Accuracy by Category

| Category | N | Initial | Deliberative | Delta |
|---|---:|---:|---:|---:|
| adversarial | 6 | 16.7% | 66.7% | +50.0% |
| direct_evidence | 6 | 100.0% | 83.3% | -16.7% |
| explicit_contradiction | 5 | 100.0% | 100.0% | +0.0% |
| implicit_contradiction | 5 | 100.0% | 100.0% | +0.0% |
| negation | 6 | 83.3% | 83.3% | +0.0% |
| over_specificity | 5 | 0.0% | 100.0% | +100.0% |
| paraphrase | 6 | 100.0% | 66.7% | -33.3% |
| partial_support | 6 | 0.0% | 33.3% | +33.3% |
| wrong_context | 5 | 20.0% | 0.0% | -20.0% |
| wrong_subject | 5 | 0.0% | 0.0% | +0.0% |

## Accuracy by Agreement Level

| Agreement | N | Initial | Deliberative | Delta |
|---|---:|---:|---:|---:|
| split | 41 | 41.5% | 56.1% | +14.6% |
| unanimous | 14 | 85.7% | 85.7% | +0.0% |

## Initial -> Final Transition Matrix

```

           CONT  PART  SUPP  UNRE
         ------------------------
 CONTRAD    23     3     0     0
 PARTIAL     0     1     0     0
 SUPPORT     1    14    12     0
 UNRELAT     1     0     0     0
```

## Case-level Results

| ID | Category | Expected | Initial | Final | Init OK | Final OK | Debate | Changed |
|---|---|---|---|---|---|---|---|---|
| d-001 | direct_evidence | SUPPORTS | SUPPORTS | PARTIAL | Y | N | Y | A,B,D |
| d-002 | direct_evidence | SUPPORTS | SUPPORTS | SUPPORTS | Y | Y | Y | A |
| d-003 | direct_evidence | SUPPORTS | SUPPORTS | SUPPORTS | Y | Y | Y | A,B,D |
| d-004 | direct_evidence | SUPPORTS | SUPPORTS | SUPPORTS | Y | Y | Y | B |
| d-005 | direct_evidence | SUPPORTS | SUPPORTS | SUPPORTS | Y | Y | Y | D |
| d-006 | direct_evidence | SUPPORTS | SUPPORTS | SUPPORTS | Y | Y | Y | A,D |
| pp-001 | paraphrase | SUPPORTS | SUPPORTS | SUPPORTS | Y | Y | Y | A |
| pp-002 | paraphrase | SUPPORTS | SUPPORTS | SUPPORTS | Y | Y | Y | A,D |
| pp-003 | paraphrase | SUPPORTS | SUPPORTS | SUPPORTS | Y | Y | Y | A,D |
| pp-004 | paraphrase | SUPPORTS | SUPPORTS | SUPPORTS | Y | Y | Y | B |
| pp-005 | paraphrase | SUPPORTS | SUPPORTS | PARTIAL | Y | N | Y | A,B,D |
| pp-006 | paraphrase | SUPPORTS | SUPPORTS | PARTIAL | Y | N | Y | A,D |
| ps-001 | partial_support | PARTIAL | SUPPORTS | CONTRADICTS | N | N | Y | A,B,D |
| ps-002 | partial_support | PARTIAL | SUPPORTS | SUPPORTS | N | N | Y | B,D |
| ps-003 | partial_support | PARTIAL | SUPPORTS | PARTIAL | N | Y | Y | A,D |
| ps-004 | partial_support | PARTIAL | SUPPORTS | SUPPORTS | N | N | Y | A,B |
| ps-005 | partial_support | PARTIAL | CONTRADICTS | PARTIAL | N | Y | Y | A,D |
| ps-006 | partial_support | PARTIAL | SUPPORTS | SUPPORTS | N | N | N | - |
| ec-001 | explicit_contradiction | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | N | - |
| ec-002 | explicit_contradiction | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | N | - |
| ec-003 | explicit_contradiction | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | N | - |
| ec-004 | explicit_contradiction | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | N | - |
| ec-005 | explicit_contradiction | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | N | - |
| ic-001 | implicit_contradiction | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | N | - |
| ic-002 | implicit_contradiction | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | Y | A,D |
| ic-003 | implicit_contradiction | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | Y | A,D |
| ic-004 | implicit_contradiction | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | N | - |
| ic-005 | implicit_contradiction | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | Y | A |
| os-001 | over_specificity | PARTIAL | SUPPORTS | PARTIAL | N | Y | Y | B,D |
| os-002 | over_specificity | PARTIAL | CONTRADICTS | PARTIAL | N | Y | Y | A,D |
| os-003 | over_specificity | PARTIAL | SUPPORTS | PARTIAL | N | Y | Y | A,C,D |
| os-004 | over_specificity | PARTIAL | SUPPORTS | PARTIAL | N | Y | Y | A,D |
| os-005 | over_specificity | PARTIAL | CONTRADICTS | PARTIAL | N | Y | Y | A,B,D |
| n-001 | negation | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | N | - |
| n-002 | negation | PARTIAL | CONTRADICTS | CONTRADICTS | N | N | Y | A,D |
| n-003 | negation | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | N | - |
| n-004 | negation | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | N | - |
| n-005 | negation | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | N | - |
| n-006 | negation | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | Y | A,B,D |
| ws-001 | wrong_subject | UNRELATED | CONTRADICTS | CONTRADICTS | N | N | Y | A,D |
| ws-002 | wrong_subject | UNRELATED | SUPPORTS | PARTIAL | N | N | Y | A |
| ws-003 | wrong_subject | UNRELATED | PARTIAL | PARTIAL | N | N | Y | A,B,D |
| ws-004 | wrong_subject | UNRELATED | SUPPORTS | PARTIAL | N | N | Y | A,D |
| ws-005 | wrong_subject | UNRELATED | SUPPORTS | PARTIAL | N | N | Y | A,B,D |
| wc-001 | wrong_context | UNRELATED | CONTRADICTS | CONTRADICTS | N | N | Y | A,B,C,D |
| wc-002 | wrong_context | UNRELATED | UNRELATED | CONTRADICTS | Y | N | Y | A,B,D |
| wc-003 | wrong_context | UNRELATED | SUPPORTS | PARTIAL | N | N | Y | A,B,D |
| wc-004 | wrong_context | UNRELATED | CONTRADICTS | CONTRADICTS | N | N | N | - |
| wc-005 | wrong_context | PARTIAL | CONTRADICTS | CONTRADICTS | N | N | Y | A,B,C,D |
| a-001 | adversarial | PARTIAL | SUPPORTS | PARTIAL | N | Y | Y | A,B,D |
| a-002 | adversarial | PARTIAL | SUPPORTS | PARTIAL | N | Y | Y | A,B,D |
| a-003 | adversarial | PARTIAL | SUPPORTS | PARTIAL | N | Y | Y | A,B,D |
| a-004 | adversarial | PARTIAL | CONTRADICTS | CONTRADICTS | N | N | Y | A,B,D |
| a-005 | adversarial | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | N | - |
| a-006 | adversarial | PARTIAL | CONTRADICTS | CONTRADICTS | N | N | Y | A,D |

## Conclusion

Mode: `debate-on-disagreement`. The deliberation produced a **net positive** effect: 10 corrections vs 4 damage (net +6). Accuracy improved from 52.7% to 63.6% (+10.9%). This provides evidence that deliberative interaction between workers can correct errors that independent voting cannot capture.
