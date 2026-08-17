# Micro-Coliseum - Deliberative Semantic Assessment

## Configuration

| Parameter | Value |
|---|---|
| Model | `qwen3-4b-rag:latest` |
| Mode | `debate-all` |
| Workers | 4 (A=entailment, B=skeptical, C=contradiction, D=context) |
| Cases | 55 |
| GPU | True |
| num_predict | 60 |
| temperature | 0.0 |
| Benchmark | semantic_assessment_benchmark_v2.json |
| Timestamp | 2026-08-16T04:50:15 |

## Accuracy

| Metric | Initial | Deliberative | Delta |
|---|---:|---:|---:|
| Accuracy | 76.4% | 74.6% | -1.8% |
| Correct | 42/55 | 41/55 | -1 |

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
| Correction rate | 30.8% |
| Damage (right->wrong) | 5 |
| Damage rate | 11.9% |
| Net effect | -1 |
| Stability rate | 83.6% |

## Debate Statistics

| Metric | Value |
|---|---:|
| Debates triggered | 55/55 |
| Debate trigger rate | 100.0% |
| Workers changed opinion | 43 |
| Revision rate | 19.6% |

## Accuracy by Category

| Category | N | Initial | Deliberative | Delta |
|---|---:|---:|---:|---:|
| adversarial | 6 | 83.3% | 83.3% | +0.0% |
| direct_evidence | 6 | 100.0% | 83.3% | -16.7% |
| explicit_contradiction | 5 | 100.0% | 100.0% | +0.0% |
| implicit_contradiction | 5 | 80.0% | 100.0% | +20.0% |
| negation | 6 | 83.3% | 66.7% | -16.7% |
| over_specificity | 5 | 100.0% | 80.0% | -20.0% |
| paraphrase | 6 | 66.7% | 66.7% | +0.0% |
| partial_support | 6 | 100.0% | 83.3% | -16.7% |
| wrong_context | 5 | 0.0% | 0.0% | +0.0% |
| wrong_subject | 5 | 40.0% | 80.0% | +40.0% |

## Accuracy by Agreement Level

| Agreement | N | Initial | Deliberative | Delta |
|---|---:|---:|---:|---:|
| split | 41 | 70.7% | 68.3% | -2.4% |
| unanimous | 14 | 92.9% | 92.9% | +0.0% |

## Initial -> Final Transition Matrix

```

           CONT  PART  SUPP  UNRE
         ------------------------
 CONTRAD    20     0     0     2
 PARTIAL     3    16     1     1
 SUPPORT     0     2     8     0
 UNRELAT     0     0     0     2
```

## Case-level Results

| ID | Category | Expected | Initial | Final | Init OK | Final OK | Debate | Changed |
|---|---|---|---|---|---|---|---|---|
| d-001 | direct_evidence | SUPPORTS | SUPPORTS | SUPPORTS | Y | Y | Y | A |
| d-002 | direct_evidence | SUPPORTS | SUPPORTS | PARTIAL | Y | N | Y | A |
| d-003 | direct_evidence | SUPPORTS | SUPPORTS | SUPPORTS | Y | Y | Y | B |
| d-004 | direct_evidence | SUPPORTS | SUPPORTS | SUPPORTS | Y | Y | Y | D |
| d-005 | direct_evidence | SUPPORTS | SUPPORTS | SUPPORTS | Y | Y | Y | C,D |
| d-006 | direct_evidence | SUPPORTS | SUPPORTS | SUPPORTS | Y | Y | Y | D |
| pp-001 | paraphrase | SUPPORTS | PARTIAL | SUPPORTS | N | Y | Y | A,C,D |
| pp-002 | paraphrase | SUPPORTS | SUPPORTS | SUPPORTS | Y | Y | Y | B,D |
| pp-003 | paraphrase | SUPPORTS | SUPPORTS | SUPPORTS | Y | Y | Y | B |
| pp-004 | paraphrase | SUPPORTS | SUPPORTS | SUPPORTS | Y | Y | Y | A,B,C |
| pp-005 | paraphrase | SUPPORTS | SUPPORTS | PARTIAL | Y | N | Y | B,D |
| pp-006 | paraphrase | SUPPORTS | PARTIAL | PARTIAL | N | N | Y | C |
| ps-001 | partial_support | PARTIAL | PARTIAL | PARTIAL | Y | Y | Y | - |
| ps-002 | partial_support | PARTIAL | PARTIAL | PARTIAL | Y | Y | Y | D |
| ps-003 | partial_support | PARTIAL | PARTIAL | PARTIAL | Y | Y | Y | D |
| ps-004 | partial_support | PARTIAL | PARTIAL | PARTIAL | Y | Y | Y | - |
| ps-005 | partial_support | PARTIAL | PARTIAL | CONTRADICTS | Y | N | Y | A,D |
| ps-006 | partial_support | PARTIAL | PARTIAL | PARTIAL | Y | Y | Y | - |
| ec-001 | explicit_contradiction | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | Y | A,D |
| ec-002 | explicit_contradiction | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | Y | - |
| ec-003 | explicit_contradiction | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | Y | A,B |
| ec-004 | explicit_contradiction | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | Y | - |
| ec-005 | explicit_contradiction | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | Y | - |
| ic-001 | implicit_contradiction | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | Y | - |
| ic-002 | implicit_contradiction | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | Y | D |
| ic-003 | implicit_contradiction | CONTRADICTS | PARTIAL | CONTRADICTS | N | Y | Y | B |
| ic-004 | implicit_contradiction | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | Y | - |
| ic-005 | implicit_contradiction | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | Y | - |
| os-001 | over_specificity | PARTIAL | PARTIAL | PARTIAL | Y | Y | Y | - |
| os-002 | over_specificity | PARTIAL | PARTIAL | PARTIAL | Y | Y | Y | D |
| os-003 | over_specificity | PARTIAL | PARTIAL | PARTIAL | Y | Y | Y | - |
| os-004 | over_specificity | PARTIAL | PARTIAL | PARTIAL | Y | Y | Y | - |
| os-005 | over_specificity | PARTIAL | PARTIAL | UNRELATED | Y | N | Y | - |
| n-001 | negation | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | Y | - |
| n-002 | negation | PARTIAL | PARTIAL | CONTRADICTS | Y | N | Y | A,C,D |
| n-003 | negation | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | Y | D |
| n-004 | negation | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | Y | - |
| n-005 | negation | CONTRADICTS | PARTIAL | PARTIAL | N | N | Y | D |
| n-006 | negation | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | Y | - |
| ws-001 | wrong_subject | UNRELATED | CONTRADICTS | UNRELATED | N | Y | Y | - |
| ws-002 | wrong_subject | UNRELATED | UNRELATED | UNRELATED | Y | Y | Y | - |
| ws-003 | wrong_subject | UNRELATED | UNRELATED | UNRELATED | Y | Y | Y | - |
| ws-004 | wrong_subject | UNRELATED | PARTIAL | PARTIAL | N | N | Y | - |
| ws-005 | wrong_subject | UNRELATED | CONTRADICTS | UNRELATED | N | Y | Y | - |
| wc-001 | wrong_context | UNRELATED | CONTRADICTS | CONTRADICTS | N | N | Y | - |
| wc-002 | wrong_context | UNRELATED | CONTRADICTS | CONTRADICTS | N | N | Y | - |
| wc-003 | wrong_context | UNRELATED | CONTRADICTS | CONTRADICTS | N | N | Y | B |
| wc-004 | wrong_context | UNRELATED | CONTRADICTS | CONTRADICTS | N | N | Y | D |
| wc-005 | wrong_context | PARTIAL | CONTRADICTS | CONTRADICTS | N | N | Y | - |
| a-001 | adversarial | PARTIAL | PARTIAL | PARTIAL | Y | Y | Y | D |
| a-002 | adversarial | PARTIAL | PARTIAL | PARTIAL | Y | Y | Y | - |
| a-003 | adversarial | PARTIAL | PARTIAL | PARTIAL | Y | Y | Y | C,D |
| a-004 | adversarial | PARTIAL | CONTRADICTS | CONTRADICTS | N | N | Y | A |
| a-005 | adversarial | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | Y | C,D |
| a-006 | adversarial | PARTIAL | PARTIAL | PARTIAL | Y | Y | Y | - |

## Conclusion

Mode: `debate-all`. The deliberation produced a **net negative** effect: 4 corrections vs 5 damage (net -1). Accuracy changed from 76.4% to 74.6% (-1.8%). The debate introduced more errors than it corrected. H0 is supported.
