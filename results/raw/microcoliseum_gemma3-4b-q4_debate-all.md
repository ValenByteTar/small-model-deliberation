# Micro-Coliseum - Deliberative Semantic Assessment

## Configuration

| Parameter | Value |
|---|---|
| Model | `gemma3:4b-it-q4_K_M` |
| Mode | `debate-all` |
| Workers | 4 (A=entailment, B=skeptical, C=contradiction, D=context) |
| Cases | 55 |
| GPU | True |
| num_predict | 60 |
| temperature | 0.0 |
| Benchmark | semantic_assessment_benchmark_v2.json |
| Timestamp | 2026-08-16T19:18:52 |

## Accuracy

| Metric | Initial | Deliberative | Delta |
|---|---:|---:|---:|
| Accuracy | 52.7% | 56.4% | +3.6% |
| Correct | 29/55 | 31/55 | +2 |

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
| Damage (right->wrong) | 8 |
| Damage rate | 27.6% |
| Net effect | +2 |
| Stability rate | 58.2% |

## Debate Statistics

| Metric | Value |
|---|---:|
| Debates triggered | 55/55 |
| Debate trigger rate | 100.0% |
| Workers changed opinion | 112 |
| Revision rate | 50.9% |

## Accuracy by Category

| Category | N | Initial | Deliberative | Delta |
|---|---:|---:|---:|---:|
| adversarial | 6 | 16.7% | 66.7% | +50.0% |
| direct_evidence | 6 | 100.0% | 83.3% | -16.7% |
| explicit_contradiction | 5 | 100.0% | 80.0% | -20.0% |
| implicit_contradiction | 5 | 100.0% | 80.0% | -20.0% |
| negation | 6 | 83.3% | 50.0% | -33.3% |
| over_specificity | 5 | 0.0% | 100.0% | +100.0% |
| paraphrase | 6 | 100.0% | 66.7% | -33.3% |
| partial_support | 6 | 0.0% | 33.3% | +33.3% |
| wrong_context | 5 | 20.0% | 0.0% | -20.0% |
| wrong_subject | 5 | 0.0% | 0.0% | +0.0% |

## Accuracy by Agreement Level

| Agreement | N | Initial | Deliberative | Delta |
|---|---:|---:|---:|---:|
| split | 41 | 41.5% | 56.1% | +14.6% |
| unanimous | 14 | 85.7% | 57.1% | -28.6% |

## Initial -> Final Transition Matrix

```

           CONT  PART  SUPP  UNRE
         ------------------------
 CONTRAD    19     7     0     0
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
| ps-006 | partial_support | PARTIAL | SUPPORTS | SUPPORTS | N | N | Y | - |
| ec-001 | explicit_contradiction | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | Y | A,C |
| ec-002 | explicit_contradiction | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | Y | A,B,C,D |
| ec-003 | explicit_contradiction | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | Y | A |
| ec-004 | explicit_contradiction | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | Y | D |
| ec-005 | explicit_contradiction | CONTRADICTS | CONTRADICTS | PARTIAL | Y | N | Y | D |
| ic-001 | implicit_contradiction | CONTRADICTS | CONTRADICTS | PARTIAL | Y | N | Y | D |
| ic-002 | implicit_contradiction | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | Y | A,D |
| ic-003 | implicit_contradiction | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | Y | A,D |
| ic-004 | implicit_contradiction | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | Y | C,D |
| ic-005 | implicit_contradiction | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | Y | A |
| os-001 | over_specificity | PARTIAL | SUPPORTS | PARTIAL | N | Y | Y | B,D |
| os-002 | over_specificity | PARTIAL | CONTRADICTS | PARTIAL | N | Y | Y | A,D |
| os-003 | over_specificity | PARTIAL | SUPPORTS | PARTIAL | N | Y | Y | A,C,D |
| os-004 | over_specificity | PARTIAL | SUPPORTS | PARTIAL | N | Y | Y | A,D |
| os-005 | over_specificity | PARTIAL | CONTRADICTS | PARTIAL | N | Y | Y | A,B,D |
| n-001 | negation | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | Y | - |
| n-002 | negation | PARTIAL | CONTRADICTS | CONTRADICTS | N | N | Y | A,D |
| n-003 | negation | CONTRADICTS | CONTRADICTS | PARTIAL | Y | N | Y | - |
| n-004 | negation | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | Y | - |
| n-005 | negation | CONTRADICTS | CONTRADICTS | PARTIAL | Y | N | Y | D |
| n-006 | negation | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | Y | A,B,D |
| ws-001 | wrong_subject | UNRELATED | CONTRADICTS | CONTRADICTS | N | N | Y | A,D |
| ws-002 | wrong_subject | UNRELATED | SUPPORTS | PARTIAL | N | N | Y | A |
| ws-003 | wrong_subject | UNRELATED | PARTIAL | PARTIAL | N | N | Y | A,B,D |
| ws-004 | wrong_subject | UNRELATED | SUPPORTS | PARTIAL | N | N | Y | A,D |
| ws-005 | wrong_subject | UNRELATED | SUPPORTS | PARTIAL | N | N | Y | A,B,D |
| wc-001 | wrong_context | UNRELATED | CONTRADICTS | CONTRADICTS | N | N | Y | A,B,C,D |
| wc-002 | wrong_context | UNRELATED | UNRELATED | CONTRADICTS | Y | N | Y | A,B,D |
| wc-003 | wrong_context | UNRELATED | SUPPORTS | PARTIAL | N | N | Y | A,B,D |
| wc-004 | wrong_context | UNRELATED | CONTRADICTS | CONTRADICTS | N | N | Y | A,B,C,D |
| wc-005 | wrong_context | PARTIAL | CONTRADICTS | CONTRADICTS | N | N | Y | A,B,C,D |
| a-001 | adversarial | PARTIAL | SUPPORTS | PARTIAL | N | Y | Y | A,B,D |
| a-002 | adversarial | PARTIAL | SUPPORTS | PARTIAL | N | Y | Y | A,B,D |
| a-003 | adversarial | PARTIAL | SUPPORTS | PARTIAL | N | Y | Y | A,B,D |
| a-004 | adversarial | PARTIAL | CONTRADICTS | CONTRADICTS | N | N | Y | A,B,D |
| a-005 | adversarial | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | Y | A |
| a-006 | adversarial | PARTIAL | CONTRADICTS | CONTRADICTS | N | N | Y | A,D |

## Conclusion

Mode: `debate-all`. The deliberation produced a **net positive** effect: 10 corrections vs 8 damage (net +2). Accuracy improved from 52.7% to 56.4% (+3.6%). This provides evidence that deliberative interaction between workers can correct errors that independent voting cannot capture.
