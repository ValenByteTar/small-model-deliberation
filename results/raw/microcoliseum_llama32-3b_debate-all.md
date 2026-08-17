# Micro-Coliseum - Deliberative Semantic Assessment

## Configuration

| Parameter | Value |
|---|---|
| Model | `llama3.2:3b` |
| Mode | `debate-all` |
| Workers | 4 (A=entailment, B=skeptical, C=contradiction, D=context) |
| Cases | 55 |
| GPU | True |
| num_predict | 64 |
| temperature | 0.0 |
| Benchmark | semantic_assessment_benchmark_v2.json |
| Timestamp | 2026-08-17T01:34:49 |

## Accuracy

| Metric | Initial | Deliberative | Delta |
|---|---:|---:|---:|
| Accuracy | 65.5% | 60.0% | -5.5% |
| Correct | 36/55 | 33/55 | -3 |

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
| Corrections (wrong->right) | 8 |
| Correction rate | 42.1% |
| Damage (right->wrong) | 11 |
| Damage rate | 30.6% |
| Net effect | -3 |
| Stability rate | 52.7% |

## Debate Statistics

| Metric | Value |
|---|---:|
| Debates triggered | 55/55 |
| Debate trigger rate | 100.0% |
| Workers changed opinion | 150 |
| Revision rate | 68.2% |

## Accuracy by Category

| Category | N | Initial | Deliberative | Delta |
|---|---:|---:|---:|---:|
| adversarial | 6 | 50.0% | 50.0% | +0.0% |
| direct_evidence | 6 | 100.0% | 100.0% | +0.0% |
| explicit_contradiction | 5 | 80.0% | 100.0% | +20.0% |
| implicit_contradiction | 5 | 80.0% | 100.0% | +20.0% |
| negation | 6 | 66.7% | 100.0% | +33.3% |
| over_specificity | 5 | 40.0% | 0.0% | -40.0% |
| paraphrase | 6 | 66.7% | 100.0% | +33.3% |
| partial_support | 6 | 33.3% | 16.7% | -16.7% |
| wrong_context | 5 | 80.0% | 20.0% | -60.0% |
| wrong_subject | 5 | 60.0% | 0.0% | -60.0% |

## Accuracy by Agreement Level

| Agreement | N | Initial | Deliberative | Delta |
|---|---:|---:|---:|---:|
| split | 48 | 62.5% | 58.3% | -4.2% |
| unanimous | 7 | 85.7% | 71.4% | -14.3% |

## Initial -> Final Transition Matrix

```

           CONT  PART  SUPP  UNRE
         ------------------------
 CONTRAD    13     0     0     0
 PARTIAL     4     1     5     0
 SUPPORT     1     1    14     0
 UNRELAT     9     2     4     1
```

## Case-level Results

| ID | Category | Expected | Initial | Final | Init OK | Final OK | Debate | Changed |
|---|---|---|---|---|---|---|---|---|
| d-001 | direct_evidence | SUPPORTS | SUPPORTS | SUPPORTS | Y | Y | Y | B |
| d-002 | direct_evidence | SUPPORTS | SUPPORTS | SUPPORTS | Y | Y | Y | - |
| d-003 | direct_evidence | SUPPORTS | SUPPORTS | SUPPORTS | Y | Y | Y | B,C |
| d-004 | direct_evidence | SUPPORTS | SUPPORTS | SUPPORTS | Y | Y | Y | C |
| d-005 | direct_evidence | SUPPORTS | SUPPORTS | SUPPORTS | Y | Y | Y | C |
| d-006 | direct_evidence | SUPPORTS | SUPPORTS | SUPPORTS | Y | Y | Y | B,C |
| pp-001 | paraphrase | SUPPORTS | UNRELATED | SUPPORTS | N | Y | Y | A,B,C,D |
| pp-002 | paraphrase | SUPPORTS | SUPPORTS | SUPPORTS | Y | Y | Y | A,D |
| pp-003 | paraphrase | SUPPORTS | SUPPORTS | SUPPORTS | Y | Y | Y | A,C,D |
| pp-004 | paraphrase | SUPPORTS | SUPPORTS | SUPPORTS | Y | Y | Y | C |
| pp-005 | paraphrase | SUPPORTS | SUPPORTS | SUPPORTS | Y | Y | Y | B |
| pp-006 | paraphrase | SUPPORTS | PARTIAL | SUPPORTS | N | Y | Y | A,C |
| ps-001 | partial_support | PARTIAL | UNRELATED | CONTRADICTS | N | N | Y | A,C |
| ps-002 | partial_support | PARTIAL | PARTIAL | SUPPORTS | Y | N | Y | A,B,C |
| ps-003 | partial_support | PARTIAL | SUPPORTS | SUPPORTS | N | N | Y | B,C |
| ps-004 | partial_support | PARTIAL | UNRELATED | PARTIAL | N | Y | Y | A,B,C,D |
| ps-005 | partial_support | PARTIAL | PARTIAL | CONTRADICTS | Y | N | Y | A,B,C,D |
| ps-006 | partial_support | PARTIAL | SUPPORTS | SUPPORTS | N | N | Y | C |
| ec-001 | explicit_contradiction | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | Y | - |
| ec-002 | explicit_contradiction | CONTRADICTS | PARTIAL | CONTRADICTS | N | Y | Y | A,B,C |
| ec-003 | explicit_contradiction | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | Y | - |
| ec-004 | explicit_contradiction | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | Y | A,B,C,D |
| ec-005 | explicit_contradiction | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | Y | - |
| ic-001 | implicit_contradiction | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | Y | A,B,C,D |
| ic-002 | implicit_contradiction | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | Y | A,B,C,D |
| ic-003 | implicit_contradiction | CONTRADICTS | UNRELATED | CONTRADICTS | N | Y | Y | A,B,C,D |
| ic-004 | implicit_contradiction | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | Y | A,B,C,D |
| ic-005 | implicit_contradiction | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | Y | A,B,C,D |
| os-001 | over_specificity | PARTIAL | UNRELATED | CONTRADICTS | N | N | Y | A,B,C,D |
| os-002 | over_specificity | PARTIAL | SUPPORTS | SUPPORTS | N | N | Y | C |
| os-003 | over_specificity | PARTIAL | PARTIAL | SUPPORTS | Y | N | Y | A,B,C,D |
| os-004 | over_specificity | PARTIAL | SUPPORTS | SUPPORTS | N | N | Y | A,B,C,D |
| os-005 | over_specificity | PARTIAL | PARTIAL | SUPPORTS | Y | N | Y | A,C |
| n-001 | negation | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | Y | A,B,C,D |
| n-002 | negation | PARTIAL | SUPPORTS | PARTIAL | N | Y | Y | A,C,D |
| n-003 | negation | CONTRADICTS | SUPPORTS | CONTRADICTS | N | Y | Y | A,B,C,D |
| n-004 | negation | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | Y | A,B,C,D |
| n-005 | negation | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | Y | A,B,C,D |
| n-006 | negation | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | Y | A,B,C,D |
| ws-001 | wrong_subject | UNRELATED | UNRELATED | SUPPORTS | Y | N | Y | A,C,D |
| ws-002 | wrong_subject | UNRELATED | PARTIAL | CONTRADICTS | N | N | Y | A,C,D |
| ws-003 | wrong_subject | UNRELATED | PARTIAL | SUPPORTS | N | N | Y | A,C |
| ws-004 | wrong_subject | UNRELATED | UNRELATED | SUPPORTS | Y | N | Y | A,C,D |
| ws-005 | wrong_subject | UNRELATED | UNRELATED | CONTRADICTS | Y | N | Y | A,C,D |
| wc-001 | wrong_context | UNRELATED | UNRELATED | CONTRADICTS | Y | N | Y | A,B,C,D |
| wc-002 | wrong_context | UNRELATED | UNRELATED | UNRELATED | Y | Y | Y | A,C |
| wc-003 | wrong_context | UNRELATED | UNRELATED | CONTRADICTS | Y | N | Y | A,B,C |
| wc-004 | wrong_context | UNRELATED | UNRELATED | CONTRADICTS | Y | N | Y | A,B,C,D |
| wc-005 | wrong_context | PARTIAL | UNRELATED | CONTRADICTS | N | N | Y | A,B,C,D |
| a-001 | adversarial | PARTIAL | PARTIAL | CONTRADICTS | Y | N | Y | A,C,D |
| a-002 | adversarial | PARTIAL | UNRELATED | SUPPORTS | N | N | Y | A,B,C,D |
| a-003 | adversarial | PARTIAL | UNRELATED | PARTIAL | N | Y | Y | B |
| a-004 | adversarial | PARTIAL | UNRELATED | CONTRADICTS | N | N | Y | B,C |
| a-005 | adversarial | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | Y | A,B,C,D |
| a-006 | adversarial | PARTIAL | PARTIAL | PARTIAL | Y | Y | Y | A,B,C,D |

## Conclusion

Mode: `debate-all`. The deliberation produced a **net negative** effect: 8 corrections vs 11 damage (net -3). Accuracy changed from 65.5% to 60.0% (-5.5%). The debate introduced more errors than it corrected. H0 is supported.
