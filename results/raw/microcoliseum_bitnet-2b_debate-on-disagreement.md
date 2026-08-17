# Micro-Coliseum - Deliberative Semantic Assessment

## Configuration

| Parameter | Value |
|---|---|
| Model | `bitnet` |
| Mode | `debate-on-disagreement` |
| Workers | 4 (A=entailment, B=skeptical, C=contradiction, D=context) |
| Cases | 55 |
| GPU | False |
| num_predict | 128 |
| temperature | 0.0 |
| Benchmark | semantic_assessment_benchmark_v2.json |
| Timestamp | 2026-08-17T00:52:28 |

## Accuracy

| Metric | Initial | Deliberative | Delta |
|---|---:|---:|---:|
| Accuracy | 0.0% | 0.0% | +0.0% |
| Correct | 0/55 | 0/55 | +0 |

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
| adversarial | 6 | 0.0% | 0.0% | +0.0% |
| direct_evidence | 6 | 0.0% | 0.0% | +0.0% |
| explicit_contradiction | 5 | 0.0% | 0.0% | +0.0% |
| implicit_contradiction | 5 | 0.0% | 0.0% | +0.0% |
| negation | 6 | 0.0% | 0.0% | +0.0% |
| over_specificity | 5 | 0.0% | 0.0% | +0.0% |
| paraphrase | 6 | 0.0% | 0.0% | +0.0% |
| partial_support | 6 | 0.0% | 0.0% | +0.0% |
| wrong_context | 5 | 0.0% | 0.0% | +0.0% |
| wrong_subject | 5 | 0.0% | 0.0% | +0.0% |

## Accuracy by Agreement Level

| Agreement | N | Initial | Deliberative | Delta |
|---|---:|---:|---:|---:|
| split | 43 | 0.0% | 0.0% | +0.0% |
| unanimous | 12 | 0.0% | 0.0% | +0.0% |

## Initial -> Final Transition Matrix

```

           CONT  PART  SUPP  UNRE
         ------------------------
 CONTRAD    12     0     0     0
 PARTIAL     0     0     0     0
 SUPPORT     0     0     0     0
 UNRELAT     0     0     0     0
```

## Case-level Results

| ID | Category | Expected | Initial | Final | Init OK | Final OK | Debate | Changed |
|---|---|---|---|---|---|---|---|---|
| d-001 | direct_evidence | SUPPORTS | CONTRADICTS | CONTRADICTS | N | N | N | - |
| d-002 | direct_evidence | SUPPORTS | CONTRADICTS | CONTRADICTS | N | N | N | - |
| d-003 | direct_evidence | SUPPORTS | CONTRADICTS | CONTRADICTS | N | N | N | - |
| d-004 | direct_evidence | SUPPORTS | CONTRADICTS | CONTRADICTS | N | N | N | - |
| d-005 | direct_evidence | SUPPORTS | CONTRADICTS | CONTRADICTS | N | N | N | - |
| d-006 | direct_evidence | SUPPORTS | CONTRADICTS | CONTRADICTS | N | N | N | - |
| pp-001 | paraphrase | SUPPORTS | CONTRADICTS | CONTRADICTS | N | N | N | - |
| pp-002 | paraphrase | SUPPORTS | CONTRADICTS | CONTRADICTS | N | N | N | - |
| pp-003 | paraphrase | SUPPORTS | CONTRADICTS | CONTRADICTS | N | N | N | - |
| pp-004 | paraphrase | SUPPORTS | CONTRADICTS | CONTRADICTS | N | N | N | - |
| pp-005 | paraphrase | SUPPORTS | CONTRADICTS | CONTRADICTS | N | N | N | - |
| pp-006 | paraphrase | SUPPORTS | CONTRADICTS | CONTRADICTS | N | N | N | - |
| ps-001 | partial_support | PARTIAL | ERROR | ERROR | N | N | N | - |
| ps-002 | partial_support | PARTIAL | ERROR | ERROR | N | N | N | - |
| ps-003 | partial_support | PARTIAL | ERROR | ERROR | N | N | N | - |
| ps-004 | partial_support | PARTIAL | ERROR | ERROR | N | N | N | - |
| ps-005 | partial_support | PARTIAL | ERROR | ERROR | N | N | N | - |
| ps-006 | partial_support | PARTIAL | ERROR | ERROR | N | N | N | - |
| ec-001 | explicit_contradiction | CONTRADICTS | ERROR | ERROR | N | N | N | - |
| ec-002 | explicit_contradiction | CONTRADICTS | ERROR | ERROR | N | N | N | - |
| ec-003 | explicit_contradiction | CONTRADICTS | ERROR | ERROR | N | N | N | - |
| ec-004 | explicit_contradiction | CONTRADICTS | ERROR | ERROR | N | N | N | - |
| ec-005 | explicit_contradiction | CONTRADICTS | ERROR | ERROR | N | N | N | - |
| ic-001 | implicit_contradiction | CONTRADICTS | ERROR | ERROR | N | N | N | - |
| ic-002 | implicit_contradiction | CONTRADICTS | ERROR | ERROR | N | N | N | - |
| ic-003 | implicit_contradiction | CONTRADICTS | ERROR | ERROR | N | N | N | - |
| ic-004 | implicit_contradiction | CONTRADICTS | ERROR | ERROR | N | N | N | - |
| ic-005 | implicit_contradiction | CONTRADICTS | ERROR | ERROR | N | N | N | - |
| os-001 | over_specificity | PARTIAL | ERROR | ERROR | N | N | N | - |
| os-002 | over_specificity | PARTIAL | ERROR | ERROR | N | N | N | - |
| os-003 | over_specificity | PARTIAL | ERROR | ERROR | N | N | N | - |
| os-004 | over_specificity | PARTIAL | ERROR | ERROR | N | N | N | - |
| os-005 | over_specificity | PARTIAL | ERROR | ERROR | N | N | N | - |
| n-001 | negation | CONTRADICTS | ERROR | ERROR | N | N | N | - |
| n-002 | negation | PARTIAL | ERROR | ERROR | N | N | N | - |
| n-003 | negation | CONTRADICTS | ERROR | ERROR | N | N | N | - |
| n-004 | negation | CONTRADICTS | ERROR | ERROR | N | N | N | - |
| n-005 | negation | CONTRADICTS | ERROR | ERROR | N | N | N | - |
| n-006 | negation | CONTRADICTS | ERROR | ERROR | N | N | N | - |
| ws-001 | wrong_subject | UNRELATED | ERROR | ERROR | N | N | N | - |
| ws-002 | wrong_subject | UNRELATED | ERROR | ERROR | N | N | N | - |
| ws-003 | wrong_subject | UNRELATED | ERROR | ERROR | N | N | N | - |
| ws-004 | wrong_subject | UNRELATED | ERROR | ERROR | N | N | N | - |
| ws-005 | wrong_subject | UNRELATED | ERROR | ERROR | N | N | N | - |
| wc-001 | wrong_context | UNRELATED | ERROR | ERROR | N | N | N | - |
| wc-002 | wrong_context | UNRELATED | ERROR | ERROR | N | N | N | - |
| wc-003 | wrong_context | UNRELATED | ERROR | ERROR | N | N | N | - |
| wc-004 | wrong_context | UNRELATED | ERROR | ERROR | N | N | N | - |
| wc-005 | wrong_context | PARTIAL | ERROR | ERROR | N | N | N | - |
| a-001 | adversarial | PARTIAL | ERROR | ERROR | N | N | N | - |
| a-002 | adversarial | PARTIAL | ERROR | ERROR | N | N | N | - |
| a-003 | adversarial | PARTIAL | ERROR | ERROR | N | N | N | - |
| a-004 | adversarial | PARTIAL | ERROR | ERROR | N | N | N | - |
| a-005 | adversarial | CONTRADICTS | ERROR | ERROR | N | N | N | - |
| a-006 | adversarial | PARTIAL | ERROR | ERROR | N | N | N | - |

## Conclusion

Mode: `debate-on-disagreement`. The deliberation had **no effect**: 0 corrections and 0 damage. Accuracy unchanged at 0.0%. The judge confirmed the initial ensemble in all cases.
