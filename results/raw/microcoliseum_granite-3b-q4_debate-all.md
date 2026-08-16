# Micro-Coliseum - Deliberative Semantic Assessment

## Configuration

| Parameter | Value |
|---|---|
| Model | `ibm/granite4.1:3b-q4_K_M` |
| Mode | `debate-all` |
| Workers | 4 (A=entailment, B=skeptical, C=contradiction, D=context) |
| Cases | 55 |
| GPU | True |
| num_predict | 60 |
| temperature | 0.0 |
| Benchmark | semantic_assessment_benchmark_v2.json |
| Timestamp | 2026-08-16T03:34:51 |

## Accuracy

| Metric | Initial | Deliberative | Delta |
|---|---:|---:|---:|
| Accuracy | 69.1% | 80.0% | +10.9% |
| Correct | 38/55 | 44/55 | +6 |

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
| Correction rate | 70.6% |
| Damage (right->wrong) | 6 |
| Damage rate | 15.8% |
| Net effect | +6 |
| Stability rate | 67.3% |

## Debate Statistics

| Metric | Value |
|---|---:|
| Debates triggered | 55/55 |
| Debate trigger rate | 100.0% |
| Workers changed opinion | 35 |
| Revision rate | 15.9% |

## Accuracy by Category

| Category | N | Initial | Deliberative | Delta |
|---|---:|---:|---:|---:|
| adversarial | 6 | 50.0% | 83.3% | +33.3% |
| direct_evidence | 6 | 83.3% | 100.0% | +16.7% |
| explicit_contradiction | 5 | 100.0% | 100.0% | +0.0% |
| implicit_contradiction | 5 | 60.0% | 80.0% | +20.0% |
| negation | 6 | 66.7% | 100.0% | +33.3% |
| over_specificity | 5 | 60.0% | 80.0% | +20.0% |
| paraphrase | 6 | 66.7% | 100.0% | +33.3% |
| partial_support | 6 | 66.7% | 83.3% | +16.7% |
| wrong_context | 5 | 60.0% | 40.0% | -20.0% |
| wrong_subject | 5 | 80.0% | 20.0% | -60.0% |

## Accuracy by Agreement Level

| Agreement | N | Initial | Deliberative | Delta |
|---|---:|---:|---:|---:|
| split | 49 | 65.3% | 77.5% | +12.2% |
| unanimous | 6 | 100.0% | 100.0% | +0.0% |

## Initial -> Final Transition Matrix

```

           CONT  PART  SUPP  UNRE
         ------------------------
 CONTRAD    12     0     0     1
 PARTIAL     1    13     0     1
 SUPPORT     0     3    12     0
 UNRELAT     3     4     5     0
```

## Case-level Results

| ID | Category | Expected | Initial | Final | Init OK | Final OK | Debate | Changed |
|---|---|---|---|---|---|---|---|---|
| d-001 | direct_evidence | SUPPORTS | SUPPORTS | SUPPORTS | Y | Y | Y | - |
| d-002 | direct_evidence | SUPPORTS | SUPPORTS | SUPPORTS | Y | Y | Y | - |
| d-003 | direct_evidence | SUPPORTS | SUPPORTS | SUPPORTS | Y | Y | Y | - |
| d-004 | direct_evidence | SUPPORTS | SUPPORTS | SUPPORTS | Y | Y | Y | - |
| d-005 | direct_evidence | SUPPORTS | UNRELATED | SUPPORTS | N | Y | Y | A,C |
| d-006 | direct_evidence | SUPPORTS | SUPPORTS | SUPPORTS | Y | Y | Y | - |
| pp-001 | paraphrase | SUPPORTS | SUPPORTS | SUPPORTS | Y | Y | Y | - |
| pp-002 | paraphrase | SUPPORTS | SUPPORTS | SUPPORTS | Y | Y | Y | - |
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
| ec-002 | explicit_contradiction | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | Y | D |
| ec-003 | explicit_contradiction | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | Y | A |
| ec-004 | explicit_contradiction | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | Y | - |
| ec-005 | explicit_contradiction | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | Y | D |
| ic-001 | implicit_contradiction | CONTRADICTS | PARTIAL | CONTRADICTS | N | Y | Y | A,B,D |
| ic-002 | implicit_contradiction | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | Y | B,C |
| ic-003 | implicit_contradiction | CONTRADICTS | PARTIAL | PARTIAL | N | N | Y | C |
| ic-004 | implicit_contradiction | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | Y | B |
| ic-005 | implicit_contradiction | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | Y | - |
| os-001 | over_specificity | PARTIAL | PARTIAL | PARTIAL | Y | Y | Y | - |
| os-002 | over_specificity | PARTIAL | UNRELATED | PARTIAL | N | Y | Y | - |
| os-003 | over_specificity | PARTIAL | PARTIAL | PARTIAL | Y | Y | Y | - |
| os-004 | over_specificity | PARTIAL | PARTIAL | PARTIAL | Y | Y | Y | C |
| os-005 | over_specificity | PARTIAL | SUPPORTS | SUPPORTS | N | N | Y | - |
| n-001 | negation | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | Y | - |
| n-002 | negation | PARTIAL | SUPPORTS | PARTIAL | N | Y | Y | - |
| n-003 | negation | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | Y | D |
| n-004 | negation | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | Y | - |
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
| wc-004 | wrong_context | UNRELATED | UNRELATED | CONTRADICTS | Y | N | Y | B,C |
| wc-005 | wrong_context | PARTIAL | PARTIAL | PARTIAL | Y | Y | Y | - |
| a-001 | adversarial | PARTIAL | SUPPORTS | PARTIAL | N | Y | Y | D |
| a-002 | adversarial | PARTIAL | SUPPORTS | SUPPORTS | N | N | Y | - |
| a-003 | adversarial | PARTIAL | PARTIAL | PARTIAL | Y | Y | Y | B |
| a-004 | adversarial | PARTIAL | PARTIAL | PARTIAL | Y | Y | Y | - |
| a-005 | adversarial | CONTRADICTS | UNRELATED | CONTRADICTS | N | Y | Y | A,B,C,D |
| a-006 | adversarial | PARTIAL | PARTIAL | PARTIAL | Y | Y | Y | - |

## Conclusion

Mode: `debate-all`. The deliberation produced a **net positive** effect: 12 corrections vs 6 damage (net +6). Accuracy improved from 69.1% to 80.0% (+10.9%). This provides evidence that deliberative interaction between workers can correct errors that independent voting cannot capture.
