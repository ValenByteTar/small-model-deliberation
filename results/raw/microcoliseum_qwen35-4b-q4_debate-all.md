# Micro-Coliseum - Deliberative Semantic Assessment

## Configuration

| Parameter | Value |
|---|---|
| Model | `qwen3.5:4b-q4_K_M` |
| Mode | `debate-all` |
| Workers | 4 (A=entailment, B=skeptical, C=contradiction, D=context) |
| Cases | 55 |
| GPU | True |
| num_predict | 60 |
| temperature | 0.0 |
| Benchmark | semantic_assessment_benchmark_v2.json |
| Timestamp | 2026-08-16T23:25:56 |

## Accuracy

| Metric | Initial | Deliberative | Delta |
|---|---:|---:|---:|
| Accuracy | 85.5% | 89.1% | +3.6% |
| Correct | 47/55 | 49/55 | +2 |

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
| Corrections (wrong->right) | 2 |
| Correction rate | 25.0% |
| Damage (right->wrong) | 0 |
| Damage rate | 0.0% |
| Net effect | +2 |
| Stability rate | 96.4% |

## Debate Statistics

| Metric | Value |
|---|---:|
| Debates triggered | 55/55 |
| Debate trigger rate | 100.0% |
| Workers changed opinion | 59 |
| Revision rate | 26.8% |

## Accuracy by Category

| Category | N | Initial | Deliberative | Delta |
|---|---:|---:|---:|---:|
| adversarial | 6 | 83.3% | 83.3% | +0.0% |
| direct_evidence | 6 | 100.0% | 100.0% | +0.0% |
| explicit_contradiction | 5 | 100.0% | 100.0% | +0.0% |
| implicit_contradiction | 5 | 100.0% | 100.0% | +0.0% |
| negation | 6 | 100.0% | 100.0% | +0.0% |
| over_specificity | 5 | 60.0% | 80.0% | +20.0% |
| paraphrase | 6 | 100.0% | 100.0% | +0.0% |
| partial_support | 6 | 66.7% | 66.7% | +0.0% |
| wrong_context | 5 | 40.0% | 60.0% | +20.0% |
| wrong_subject | 5 | 100.0% | 100.0% | +0.0% |

## Accuracy by Agreement Level

| Agreement | N | Initial | Deliberative | Delta |
|---|---:|---:|---:|---:|
| split | 23 | 78.3% | 87.0% | +8.7% |
| unanimous | 32 | 90.6% | 90.6% | +0.0% |

## Initial -> Final Transition Matrix

```

           CONT  PART  SUPP  UNRE
         ------------------------
 CONTRAD    18     0     0     1
 PARTIAL     0    13     0     0
 SUPPORT     0     1    15     0
 UNRELAT     0     0     0     7
```

## Case-level Results

| ID | Category | Expected | Initial | Final | Init OK | Final OK | Debate | Changed |
|---|---|---|---|---|---|---|---|---|
| d-001 | direct_evidence | SUPPORTS | SUPPORTS | SUPPORTS | Y | Y | Y | - |
| d-002 | direct_evidence | SUPPORTS | SUPPORTS | SUPPORTS | Y | Y | Y | - |
| d-003 | direct_evidence | SUPPORTS | SUPPORTS | SUPPORTS | Y | Y | Y | - |
| d-004 | direct_evidence | SUPPORTS | SUPPORTS | SUPPORTS | Y | Y | Y | - |
| d-005 | direct_evidence | SUPPORTS | SUPPORTS | SUPPORTS | Y | Y | Y | - |
| d-006 | direct_evidence | SUPPORTS | SUPPORTS | SUPPORTS | Y | Y | Y | - |
| pp-001 | paraphrase | SUPPORTS | SUPPORTS | SUPPORTS | Y | Y | Y | - |
| pp-002 | paraphrase | SUPPORTS | SUPPORTS | SUPPORTS | Y | Y | Y | - |
| pp-003 | paraphrase | SUPPORTS | SUPPORTS | SUPPORTS | Y | Y | Y | - |
| pp-004 | paraphrase | SUPPORTS | SUPPORTS | SUPPORTS | Y | Y | Y | B |
| pp-005 | paraphrase | SUPPORTS | SUPPORTS | SUPPORTS | Y | Y | Y | - |
| pp-006 | paraphrase | SUPPORTS | SUPPORTS | SUPPORTS | Y | Y | Y | B |
| ps-001 | partial_support | PARTIAL | PARTIAL | PARTIAL | Y | Y | Y | - |
| ps-002 | partial_support | PARTIAL | PARTIAL | PARTIAL | Y | Y | Y | A,B,C,D |
| ps-003 | partial_support | PARTIAL | SUPPORTS | SUPPORTS | N | N | Y | - |
| ps-004 | partial_support | PARTIAL | PARTIAL | PARTIAL | Y | Y | Y | A |
| ps-005 | partial_support | PARTIAL | PARTIAL | PARTIAL | Y | Y | Y | - |
| ps-006 | partial_support | PARTIAL | SUPPORTS | SUPPORTS | N | N | Y | A,B |
| ec-001 | explicit_contradiction | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | Y | - |
| ec-002 | explicit_contradiction | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | Y | C,D |
| ec-003 | explicit_contradiction | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | Y | - |
| ec-004 | explicit_contradiction | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | Y | - |
| ec-005 | explicit_contradiction | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | Y | - |
| ic-001 | implicit_contradiction | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | Y | - |
| ic-002 | implicit_contradiction | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | Y | - |
| ic-003 | implicit_contradiction | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | Y | A |
| ic-004 | implicit_contradiction | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | Y | - |
| ic-005 | implicit_contradiction | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | Y | - |
| os-001 | over_specificity | PARTIAL | PARTIAL | PARTIAL | Y | Y | Y | - |
| os-002 | over_specificity | PARTIAL | SUPPORTS | SUPPORTS | N | N | Y | C |
| os-003 | over_specificity | PARTIAL | PARTIAL | PARTIAL | Y | Y | Y | A,D |
| os-004 | over_specificity | PARTIAL | SUPPORTS | PARTIAL | N | Y | Y | A,C |
| os-005 | over_specificity | PARTIAL | PARTIAL | PARTIAL | Y | Y | Y | A,C |
| n-001 | negation | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | Y | - |
| n-002 | negation | PARTIAL | PARTIAL | PARTIAL | Y | Y | Y | B |
| n-003 | negation | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | Y | - |
| n-004 | negation | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | Y | - |
| n-005 | negation | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | Y | C |
| n-006 | negation | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | Y | A,B |
| ws-001 | wrong_subject | UNRELATED | UNRELATED | UNRELATED | Y | Y | Y | A |
| ws-002 | wrong_subject | UNRELATED | UNRELATED | UNRELATED | Y | Y | Y | A,B,C,D |
| ws-003 | wrong_subject | UNRELATED | UNRELATED | UNRELATED | Y | Y | Y | A,B,C,D |
| ws-004 | wrong_subject | UNRELATED | UNRELATED | UNRELATED | Y | Y | Y | A,B,C,D |
| ws-005 | wrong_subject | UNRELATED | UNRELATED | UNRELATED | Y | Y | Y | A,B,C |
| wc-001 | wrong_context | UNRELATED | CONTRADICTS | UNRELATED | N | Y | Y | A |
| wc-002 | wrong_context | UNRELATED | UNRELATED | UNRELATED | Y | Y | Y | B,C,D |
| wc-003 | wrong_context | UNRELATED | CONTRADICTS | CONTRADICTS | N | N | Y | B,C,D |
| wc-004 | wrong_context | UNRELATED | CONTRADICTS | CONTRADICTS | N | N | Y | - |
| wc-005 | wrong_context | PARTIAL | PARTIAL | PARTIAL | Y | Y | Y | A,C,D |
| a-001 | adversarial | PARTIAL | PARTIAL | PARTIAL | Y | Y | Y | A |
| a-002 | adversarial | PARTIAL | UNRELATED | UNRELATED | N | N | Y | A,B,C,D |
| a-003 | adversarial | PARTIAL | PARTIAL | PARTIAL | Y | Y | Y | A,C,D |
| a-004 | adversarial | PARTIAL | PARTIAL | PARTIAL | Y | Y | Y | D |
| a-005 | adversarial | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | Y | - |
| a-006 | adversarial | PARTIAL | PARTIAL | PARTIAL | Y | Y | Y | C |

## Conclusion

Mode: `debate-all`. The deliberation produced a **net positive** effect: 2 corrections vs 0 damage (net +2). Accuracy improved from 85.5% to 89.1% (+3.6%). This provides evidence that deliberative interaction between workers can correct errors that independent voting cannot capture.
