# Micro-Coliseum - Deliberative Semantic Assessment

## Configuration

| Parameter | Value |
|---|---|
| Model | `dhiltgen/nemotron-3-nano:4b` |
| Mode | `debate-on-disagreement` |
| Workers | 4 (A=entailment, B=skeptical, C=contradiction, D=context) |
| Cases | 55 |
| GPU | True |
| num_predict | 60 |
| temperature | 0.0 |
| Benchmark | semantic_assessment_benchmark_v2.json |
| Timestamp | 2026-08-16T20:18:33 |

## Accuracy

| Metric | Initial | Deliberative | Delta |
|---|---:|---:|---:|
| Accuracy | 60.0% | 61.8% | +1.8% |
| Correct | 33/55 | 34/55 | +1 |

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
| Corrections (wrong->right) | 3 |
| Correction rate | 13.6% |
| Damage (right->wrong) | 2 |
| Damage rate | 6.1% |
| Net effect | +1 |
| Stability rate | 87.3% |

## Debate Statistics

| Metric | Value |
|---|---:|
| Debates triggered | 23/55 |
| Debate trigger rate | 41.8% |
| Workers changed opinion | 5 |
| Revision rate | 5.4% |

## Accuracy by Category

| Category | N | Initial | Deliberative | Delta |
|---|---:|---:|---:|---:|
| adversarial | 6 | 16.7% | 33.3% | +16.7% |
| direct_evidence | 6 | 100.0% | 100.0% | +0.0% |
| explicit_contradiction | 5 | 100.0% | 100.0% | +0.0% |
| implicit_contradiction | 5 | 100.0% | 100.0% | +0.0% |
| negation | 6 | 66.7% | 100.0% | +33.3% |
| over_specificity | 5 | 0.0% | 0.0% | +0.0% |
| paraphrase | 6 | 100.0% | 100.0% | +0.0% |
| partial_support | 6 | 0.0% | 0.0% | +0.0% |
| wrong_context | 5 | 40.0% | 0.0% | -40.0% |
| wrong_subject | 5 | 80.0% | 80.0% | +0.0% |

## Accuracy by Agreement Level

| Agreement | N | Initial | Deliberative | Delta |
|---|---:|---:|---:|---:|
| split | 23 | 13.0% | 17.4% | +4.4% |
| unanimous | 32 | 93.8% | 93.8% | +0.0% |

## Initial -> Final Transition Matrix

```

           CONT  PART  SUPP  UNRE
         ------------------------
 CONTRAD    20     0     0     0
 PARTIAL     0     0     0     0
 SUPPORT     3     2    24     0
 UNRELAT     2     0     0     4
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
| pp-006 | paraphrase | SUPPORTS | SUPPORTS | SUPPORTS | Y | Y | N | - |
| ps-001 | partial_support | PARTIAL | SUPPORTS | SUPPORTS | N | N | Y | - |
| ps-002 | partial_support | PARTIAL | SUPPORTS | SUPPORTS | N | N | Y | - |
| ps-003 | partial_support | PARTIAL | SUPPORTS | SUPPORTS | N | N | Y | B |
| ps-004 | partial_support | PARTIAL | SUPPORTS | SUPPORTS | N | N | Y | - |
| ps-005 | partial_support | PARTIAL | SUPPORTS | CONTRADICTS | N | N | Y | - |
| ps-006 | partial_support | PARTIAL | SUPPORTS | SUPPORTS | N | N | N | - |
| ec-001 | explicit_contradiction | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | N | - |
| ec-002 | explicit_contradiction | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | N | - |
| ec-003 | explicit_contradiction | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | N | - |
| ec-004 | explicit_contradiction | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | N | - |
| ec-005 | explicit_contradiction | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | N | - |
| ic-001 | implicit_contradiction | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | N | - |
| ic-002 | implicit_contradiction | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | N | - |
| ic-003 | implicit_contradiction | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | Y | - |
| ic-004 | implicit_contradiction | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | N | - |
| ic-005 | implicit_contradiction | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | N | - |
| os-001 | over_specificity | PARTIAL | SUPPORTS | SUPPORTS | N | N | Y | - |
| os-002 | over_specificity | PARTIAL | SUPPORTS | SUPPORTS | N | N | Y | - |
| os-003 | over_specificity | PARTIAL | SUPPORTS | SUPPORTS | N | N | Y | - |
| os-004 | over_specificity | PARTIAL | SUPPORTS | SUPPORTS | N | N | Y | - |
| os-005 | over_specificity | PARTIAL | SUPPORTS | SUPPORTS | N | N | Y | - |
| n-001 | negation | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | N | - |
| n-002 | negation | PARTIAL | SUPPORTS | PARTIAL | N | Y | Y | - |
| n-003 | negation | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | N | - |
| n-004 | negation | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | N | - |
| n-005 | negation | CONTRADICTS | SUPPORTS | CONTRADICTS | N | Y | Y | - |
| n-006 | negation | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | N | - |
| ws-001 | wrong_subject | UNRELATED | UNRELATED | UNRELATED | Y | Y | N | - |
| ws-002 | wrong_subject | UNRELATED | UNRELATED | UNRELATED | Y | Y | N | - |
| ws-003 | wrong_subject | UNRELATED | UNRELATED | UNRELATED | Y | Y | N | - |
| ws-004 | wrong_subject | UNRELATED | UNRELATED | UNRELATED | Y | Y | N | - |
| ws-005 | wrong_subject | UNRELATED | SUPPORTS | SUPPORTS | N | N | Y | B,D |
| wc-001 | wrong_context | UNRELATED | UNRELATED | CONTRADICTS | Y | N | Y | - |
| wc-002 | wrong_context | UNRELATED | UNRELATED | CONTRADICTS | Y | N | Y | - |
| wc-003 | wrong_context | UNRELATED | SUPPORTS | CONTRADICTS | N | N | Y | - |
| wc-004 | wrong_context | UNRELATED | CONTRADICTS | CONTRADICTS | N | N | N | - |
| wc-005 | wrong_context | PARTIAL | CONTRADICTS | CONTRADICTS | N | N | Y | - |
| a-001 | adversarial | PARTIAL | SUPPORTS | SUPPORTS | N | N | Y | B |
| a-002 | adversarial | PARTIAL | CONTRADICTS | CONTRADICTS | N | N | Y | - |
| a-003 | adversarial | PARTIAL | SUPPORTS | PARTIAL | N | Y | Y | - |
| a-004 | adversarial | PARTIAL | CONTRADICTS | CONTRADICTS | N | N | Y | - |
| a-005 | adversarial | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | N | - |
| a-006 | adversarial | PARTIAL | CONTRADICTS | CONTRADICTS | N | N | Y | A |

## Conclusion

Mode: `debate-on-disagreement`. The deliberation produced a **net positive** effect: 3 corrections vs 2 damage (net +1). Accuracy improved from 60.0% to 61.8% (+1.8%). This provides evidence that deliberative interaction between workers can correct errors that independent voting cannot capture.
