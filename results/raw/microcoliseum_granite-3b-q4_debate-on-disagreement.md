# Micro-Coliseum - Deliberative Semantic Assessment

## Configuration

| Parameter | Value |
|---|---|
| Model | `ibm/granite4.1:3b-q4_K_M` |
| Mode | `debate-on-disagreement` |
| Workers | 4 (A=entailment, B=skeptical, C=contradiction, D=context) |
| Cases | 55 |
| GPU | True |
| num_predict | 60 |
| temperature | 0.0 |
| Benchmark | semantic_assessment_benchmark_v2.json |
| Timestamp | 2026-08-16T03:05:42 |

## Accuracy

| Metric | Initial | Deliberative | Delta |
|---|---:|---:|---:|
| Accuracy | 67.3% | 69.1% | +1.8% |
| Correct | 37/55 | 38/55 | +1 |

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
| Corrections (wrong->right) | 12 |
| Correction rate | 66.7% |
| Damage (right->wrong) | 11 |
| Damage rate | 29.7% |
| Net effect | +1 |
| Stability rate | 56.4% |

## Debate Statistics

| Metric | Value |
|---|---:|
| Debates triggered | 49/55 |
| Debate trigger rate | 89.1% |
| Workers changed opinion | 31 |
| Revision rate | 15.8% |

## Accuracy by Category

| Category | N | Initial | Deliberative | Delta |
|---|---:|---:|---:|---:|
| adversarial | 6 | 50.0% | 66.7% | +16.7% |
| direct_evidence | 6 | 83.3% | 83.3% | +0.0% |
| explicit_contradiction | 5 | 100.0% | 100.0% | +0.0% |
| implicit_contradiction | 5 | 60.0% | 80.0% | +20.0% |
| negation | 6 | 66.7% | 66.7% | +0.0% |
| over_specificity | 5 | 60.0% | 80.0% | +20.0% |
| paraphrase | 6 | 66.7% | 66.7% | +0.0% |
| partial_support | 6 | 66.7% | 83.3% | +16.7% |
| wrong_context | 5 | 40.0% | 40.0% | +0.0% |
| wrong_subject | 5 | 80.0% | 20.0% | -60.0% |

## Accuracy by Agreement Level

| Agreement | N | Initial | Deliberative | Delta |
|---|---:|---:|---:|---:|
| split | 49 | 63.3% | 77.5% | +14.3% |
| unanimous | 6 | 100.0% | 0.0% | -100.0% |

## Initial -> Final Transition Matrix

```

           CONT  PART  SUPP  UNRE
         ------------------------
 CONTRAD    10     0     0     1
 PARTIAL     2    12     0     1
 SUPPORT     0     3     9     0
 UNRELAT     2     4     5     0
```

## Case-level Results

| ID | Category | Expected | Initial | Final | Init OK | Final OK | Debate | Changed |
|---|---|---|---|---|---|---|---|---|
| d-001 | direct_evidence | SUPPORTS | SUPPORTS | SUPPORTS | Y | Y | Y | - |
| d-002 | direct_evidence | SUPPORTS | SUPPORTS | SUPPORTS | Y | Y | Y | - |
| d-003 | direct_evidence | SUPPORTS | SUPPORTS | SUPPORTS | Y | Y | Y | - |
| d-004 | direct_evidence | SUPPORTS | SUPPORTS |  | Y | N | N | - |
| d-005 | direct_evidence | SUPPORTS | UNRELATED | SUPPORTS | N | Y | Y | A,C |
| d-006 | direct_evidence | SUPPORTS | SUPPORTS | SUPPORTS | Y | Y | Y | - |
| pp-001 | paraphrase | SUPPORTS | SUPPORTS |  | Y | N | N | - |
| pp-002 | paraphrase | SUPPORTS | SUPPORTS |  | Y | N | N | - |
| pp-003 | paraphrase | SUPPORTS | SUPPORTS | SUPPORTS | Y | Y | Y | B |
| pp-004 | paraphrase | SUPPORTS | UNRELATED | SUPPORTS | N | Y | Y | A,B |
| pp-005 | paraphrase | SUPPORTS | SUPPORTS | SUPPORTS | Y | Y | Y | - |
| pp-006 | paraphrase | SUPPORTS | UNRELATED | SUPPORTS | N | Y | Y | D |
| ps-001 | partial_support | PARTIAL | PARTIAL | PARTIAL | Y | Y | Y | - |
| ps-002 | partial_support | PARTIAL | PARTIAL | PARTIAL | Y | Y | Y | - |
| ps-003 | partial_support | PARTIAL | SUPPORTS | SUPPORTS | N | N | Y | - |
| ps-004 | partial_support | PARTIAL | PARTIAL | PARTIAL | Y | Y | Y | - |
| ps-005 | partial_support | PARTIAL | PARTIAL | PARTIAL | Y | Y | Y | - |
| ps-006 | partial_support | PARTIAL | SUPPORTS | PARTIAL | N | Y | Y | - |
| ec-001 | explicit_contradiction | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | Y | A,D |
| ec-002 | explicit_contradiction | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | Y | B |
| ec-003 | explicit_contradiction | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | Y | A |
| ec-004 | explicit_contradiction | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | Y | - |
| ec-005 | explicit_contradiction | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | Y | D |
| ic-001 | implicit_contradiction | CONTRADICTS | PARTIAL | CONTRADICTS | N | Y | Y | - |
| ic-002 | implicit_contradiction | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | Y | B,C |
| ic-003 | implicit_contradiction | CONTRADICTS | PARTIAL | PARTIAL | N | N | Y | C |
| ic-004 | implicit_contradiction | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | Y | B |
| ic-005 | implicit_contradiction | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | Y | - |
| os-001 | over_specificity | PARTIAL | PARTIAL | PARTIAL | Y | Y | Y | - |
| os-002 | over_specificity | PARTIAL | UNRELATED | PARTIAL | N | Y | Y | - |
| os-003 | over_specificity | PARTIAL | PARTIAL | PARTIAL | Y | Y | Y | - |
| os-004 | over_specificity | PARTIAL | PARTIAL | PARTIAL | Y | Y | Y | C |
| os-005 | over_specificity | PARTIAL | SUPPORTS | SUPPORTS | N | N | Y | - |
| n-001 | negation | CONTRADICTS | CONTRADICTS |  | Y | N | N | - |
| n-002 | negation | PARTIAL | SUPPORTS | PARTIAL | N | Y | Y | - |
| n-003 | negation | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | Y | D |
| n-004 | negation | CONTRADICTS | CONTRADICTS |  | Y | N | N | - |
| n-005 | negation | CONTRADICTS | UNRELATED | CONTRADICTS | N | Y | Y | B |
| n-006 | negation | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | Y | D |
| ws-001 | wrong_subject | UNRELATED | PARTIAL | UNRELATED | N | Y | Y | - |
| ws-002 | wrong_subject | UNRELATED | UNRELATED | SUPPORTS | Y | N | Y | - |
| ws-003 | wrong_subject | UNRELATED | UNRELATED | PARTIAL | Y | N | Y | - |
| ws-004 | wrong_subject | UNRELATED | UNRELATED | SUPPORTS | Y | N | Y | A,D |
| ws-005 | wrong_subject | UNRELATED | UNRELATED | PARTIAL | Y | N | Y | B,C,D |
| wc-001 | wrong_context | UNRELATED | PARTIAL | PARTIAL | N | N | Y | - |
| wc-002 | wrong_context | UNRELATED | CONTRADICTS | UNRELATED | N | Y | Y | - |
| wc-003 | wrong_context | UNRELATED | UNRELATED | PARTIAL | Y | N | Y | - |
| wc-004 | wrong_context | UNRELATED | PARTIAL | CONTRADICTS | N | N | Y | C |
| wc-005 | wrong_context | PARTIAL | PARTIAL | PARTIAL | Y | Y | Y | - |
| a-001 | adversarial | PARTIAL | SUPPORTS | PARTIAL | N | Y | Y | D |
| a-002 | adversarial | PARTIAL | SUPPORTS | SUPPORTS | N | N | Y | - |
| a-003 | adversarial | PARTIAL | PARTIAL | PARTIAL | Y | Y | Y | B |
| a-004 | adversarial | PARTIAL | PARTIAL | PARTIAL | Y | Y | Y | - |
| a-005 | adversarial | CONTRADICTS | UNRELATED | CONTRADICTS | N | Y | Y | A,B,C,D |
| a-006 | adversarial | PARTIAL | PARTIAL |  | Y | N | N | - |

## Conclusion

Mode: `debate-on-disagreement`. The deliberation produced a **net positive** effect: 12 corrections vs 11 damage (net +1). Accuracy improved from 67.3% to 69.1% (+1.8%). This provides evidence that deliberative interaction between workers can correct errors that independent voting cannot capture.
