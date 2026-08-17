# Micro-Coliseum - Deliberative Semantic Assessment

## Configuration

| Parameter | Value |
|---|---|
| Model | `TechyShishy/ministral-3:3b-reasoning-2512-q4_K_M` |
| Mode | `debate-on-disagreement` |
| Workers | 4 (A=entailment, B=skeptical, C=contradiction, D=context) |
| Cases | 55 |
| GPU | True |
| num_predict | 60 |
| temperature | 0.0 |
| Benchmark | semantic_assessment_benchmark_v2.json |
| Timestamp | 2026-08-16T21:24:09 |

## Accuracy

| Metric | Initial | Deliberative | Delta |
|---|---:|---:|---:|
| Accuracy | 74.6% | 60.0% | -14.5% |
| Correct | 41/55 | 33/55 | -8 |

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
| Corrections (wrong->right) | 4 |
| Correction rate | 28.6% |
| Damage (right->wrong) | 12 |
| Damage rate | 29.3% |
| Net effect | -8 |
| Stability rate | 65.5% |

## Debate Statistics

| Metric | Value |
|---|---:|
| Debates triggered | 34/55 |
| Debate trigger rate | 61.8% |
| Workers changed opinion | 0 |
| Revision rate | 0.0% |

## Accuracy by Category

| Category | N | Initial | Deliberative | Delta |
|---|---:|---:|---:|---:|
| adversarial | 6 | 66.7% | 16.7% | -50.0% |
| direct_evidence | 6 | 83.3% | 100.0% | +16.7% |
| explicit_contradiction | 5 | 100.0% | 100.0% | +0.0% |
| implicit_contradiction | 5 | 100.0% | 100.0% | +0.0% |
| negation | 6 | 100.0% | 100.0% | +0.0% |
| over_specificity | 5 | 80.0% | 0.0% | -80.0% |
| paraphrase | 6 | 100.0% | 100.0% | +0.0% |
| partial_support | 6 | 83.3% | 0.0% | -83.3% |
| wrong_context | 5 | 0.0% | 20.0% | +20.0% |
| wrong_subject | 5 | 20.0% | 60.0% | +40.0% |

## Accuracy by Agreement Level

| Agreement | N | Initial | Deliberative | Delta |
|---|---:|---:|---:|---:|
| split | 34 | 61.8% | 38.2% | -23.5% |
| unanimous | 21 | 95.2% | 95.2% | +0.0% |

## Initial -> Final Transition Matrix

```

           CONT  PART  SUPP  UNRE
         ------------------------
 CONTRAD    18     0     3     4
 PARTIAL     0     1    12     0
 SUPPORT     0     0    16     0
 UNRELAT     0     0     0     1
```

## Case-level Results

| ID | Category | Expected | Initial | Final | Init OK | Final OK | Debate | Changed |
|---|---|---|---|---|---|---|---|---|
| d-001 | direct_evidence | SUPPORTS | CONTRADICTS | SUPPORTS | N | Y | Y | - |
| d-002 | direct_evidence | SUPPORTS | SUPPORTS | SUPPORTS | Y | Y | N | - |
| d-003 | direct_evidence | SUPPORTS | SUPPORTS | SUPPORTS | Y | Y | N | - |
| d-004 | direct_evidence | SUPPORTS | SUPPORTS | SUPPORTS | Y | Y | N | - |
| d-005 | direct_evidence | SUPPORTS | SUPPORTS | SUPPORTS | Y | Y | N | - |
| d-006 | direct_evidence | SUPPORTS | SUPPORTS | SUPPORTS | Y | Y | Y | - |
| pp-001 | paraphrase | SUPPORTS | SUPPORTS | SUPPORTS | Y | Y | Y | - |
| pp-002 | paraphrase | SUPPORTS | SUPPORTS | SUPPORTS | Y | Y | Y | - |
| pp-003 | paraphrase | SUPPORTS | SUPPORTS | SUPPORTS | Y | Y | N | - |
| pp-004 | paraphrase | SUPPORTS | SUPPORTS | SUPPORTS | Y | Y | Y | - |
| pp-005 | paraphrase | SUPPORTS | SUPPORTS | SUPPORTS | Y | Y | N | - |
| pp-006 | paraphrase | SUPPORTS | SUPPORTS | SUPPORTS | Y | Y | Y | - |
| ps-001 | partial_support | PARTIAL | PARTIAL | SUPPORTS | Y | N | Y | - |
| ps-002 | partial_support | PARTIAL | PARTIAL | SUPPORTS | Y | N | Y | - |
| ps-003 | partial_support | PARTIAL | SUPPORTS | SUPPORTS | N | N | Y | - |
| ps-004 | partial_support | PARTIAL | PARTIAL | SUPPORTS | Y | N | Y | - |
| ps-005 | partial_support | PARTIAL | PARTIAL | SUPPORTS | Y | N | Y | - |
| ps-006 | partial_support | PARTIAL | PARTIAL | SUPPORTS | Y | N | Y | - |
| ec-001 | explicit_contradiction | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | N | - |
| ec-002 | explicit_contradiction | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | N | - |
| ec-003 | explicit_contradiction | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | N | - |
| ec-004 | explicit_contradiction | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | N | - |
| ec-005 | explicit_contradiction | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | N | - |
| ic-001 | implicit_contradiction | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | N | - |
| ic-002 | implicit_contradiction | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | N | - |
| ic-003 | implicit_contradiction | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | Y | - |
| ic-004 | implicit_contradiction | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | N | - |
| ic-005 | implicit_contradiction | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | Y | - |
| os-001 | over_specificity | PARTIAL | PARTIAL | SUPPORTS | Y | N | Y | - |
| os-002 | over_specificity | PARTIAL | SUPPORTS | SUPPORTS | N | N | N | - |
| os-003 | over_specificity | PARTIAL | PARTIAL | SUPPORTS | Y | N | Y | - |
| os-004 | over_specificity | PARTIAL | PARTIAL | SUPPORTS | Y | N | Y | - |
| os-005 | over_specificity | PARTIAL | PARTIAL | SUPPORTS | Y | N | Y | - |
| n-001 | negation | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | Y | - |
| n-002 | negation | PARTIAL | PARTIAL | PARTIAL | Y | Y | Y | - |
| n-003 | negation | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | N | - |
| n-004 | negation | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | N | - |
| n-005 | negation | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | N | - |
| n-006 | negation | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | N | - |
| ws-001 | wrong_subject | UNRELATED | UNRELATED | UNRELATED | Y | Y | N | - |
| ws-002 | wrong_subject | UNRELATED | CONTRADICTS | UNRELATED | N | Y | Y | - |
| ws-003 | wrong_subject | UNRELATED | CONTRADICTS | UNRELATED | N | Y | Y | - |
| ws-004 | wrong_subject | UNRELATED | SUPPORTS | SUPPORTS | N | N | Y | - |
| ws-005 | wrong_subject | UNRELATED | CONTRADICTS | SUPPORTS | N | N | Y | - |
| wc-001 | wrong_context | UNRELATED | CONTRADICTS | CONTRADICTS | N | N | Y | - |
| wc-002 | wrong_context | UNRELATED | CONTRADICTS | UNRELATED | N | Y | Y | - |
| wc-003 | wrong_context | UNRELATED | CONTRADICTS | SUPPORTS | N | N | Y | - |
| wc-004 | wrong_context | UNRELATED | CONTRADICTS | CONTRADICTS | N | N | Y | - |
| wc-005 | wrong_context | PARTIAL | CONTRADICTS | UNRELATED | N | N | Y | - |
| a-001 | adversarial | PARTIAL | SUPPORTS | SUPPORTS | N | N | Y | - |
| a-002 | adversarial | PARTIAL | PARTIAL | SUPPORTS | Y | N | Y | - |
| a-003 | adversarial | PARTIAL | SUPPORTS | SUPPORTS | N | N | Y | - |
| a-004 | adversarial | PARTIAL | PARTIAL | SUPPORTS | Y | N | Y | - |
| a-005 | adversarial | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | N | - |
| a-006 | adversarial | PARTIAL | PARTIAL | SUPPORTS | Y | N | Y | - |

## Conclusion

Mode: `debate-on-disagreement`. The deliberation produced a **net negative** effect: 4 corrections vs 12 damage (net -8). Accuracy changed from 74.6% to 60.0% (-14.5%). The debate introduced more errors than it corrected. H0 is supported.
