# Micro-Coliseum - Deliberative Semantic Assessment

## Configuration

| Parameter | Value |
|---|---|
| Model | `hf.co/Qwen/Qwen3-4B-GGUF:Q4_K_M` |
| Mode | `debate-on-disagreement` |
| Workers | 4 (A=entailment, B=skeptical, C=contradiction, D=context) |
| Cases | 55 |
| GPU | True |
| num_predict | 64 |
| temperature | 0.0 |
| Benchmark | semantic_assessment_benchmark_v2.json |
| Timestamp | 2026-08-17T02:16:19 |

## Accuracy

| Metric | Initial | Deliberative | Delta |
|---|---:|---:|---:|
| Accuracy | 65.5% | 54.5% | -10.9% |
| Correct | 36/55 | 30/55 | -6 |

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
| Correction rate | 10.5% |
| Damage (right->wrong) | 8 |
| Damage rate | 22.2% |
| Net effect | -6 |
| Stability rate | 74.6% |

## Debate Statistics

| Metric | Value |
|---|---:|
| Debates triggered | 39/55 |
| Debate trigger rate | 70.9% |
| Workers changed opinion | 43 |
| Revision rate | 27.6% |

## Accuracy by Category

| Category | N | Initial | Deliberative | Delta |
|---|---:|---:|---:|---:|
| adversarial | 6 | 50.0% | 16.7% | -33.3% |
| direct_evidence | 6 | 100.0% | 100.0% | +0.0% |
| explicit_contradiction | 5 | 100.0% | 100.0% | +0.0% |
| implicit_contradiction | 5 | 100.0% | 100.0% | +0.0% |
| negation | 6 | 83.3% | 100.0% | +16.7% |
| over_specificity | 5 | 40.0% | 0.0% | -40.0% |
| paraphrase | 6 | 83.3% | 100.0% | +16.7% |
| partial_support | 6 | 33.3% | 16.7% | -16.7% |
| wrong_context | 5 | 0.0% | 0.0% | +0.0% |
| wrong_subject | 5 | 60.0% | 0.0% | -60.0% |

## Accuracy by Agreement Level

| Agreement | N | Initial | Deliberative | Delta |
|---|---:|---:|---:|---:|
| split | 39 | 53.8% | 38.5% | -15.4% |
| unanimous | 16 | 93.8% | 93.8% | +0.0% |

## Initial -> Final Transition Matrix

```

           CONT  PART  SUPP  UNRE
         ------------------------
 CONTRAD    24     1     4     0
 PARTIAL     0     1     6     0
 SUPPORT     0     0    16     0
 UNRELAT     1     2     0     0
```

## Case-level Results

| ID | Category | Expected | Initial | Final | Init OK | Final OK | Debate | Changed |
|---|---|---|---|---|---|---|---|---|
| d-001 | direct_evidence | SUPPORTS | SUPPORTS | SUPPORTS | Y | Y | Y | B |
| d-002 | direct_evidence | SUPPORTS | SUPPORTS | SUPPORTS | Y | Y | Y | - |
| d-003 | direct_evidence | SUPPORTS | SUPPORTS | SUPPORTS | Y | Y | Y | B |
| d-004 | direct_evidence | SUPPORTS | SUPPORTS | SUPPORTS | Y | Y | Y | - |
| d-005 | direct_evidence | SUPPORTS | SUPPORTS | SUPPORTS | Y | Y | Y | - |
| d-006 | direct_evidence | SUPPORTS | SUPPORTS | SUPPORTS | Y | Y | Y | - |
| pp-001 | paraphrase | SUPPORTS | SUPPORTS | SUPPORTS | Y | Y | Y | - |
| pp-002 | paraphrase | SUPPORTS | SUPPORTS | SUPPORTS | Y | Y | Y | B |
| pp-003 | paraphrase | SUPPORTS | SUPPORTS | SUPPORTS | Y | Y | Y | - |
| pp-004 | paraphrase | SUPPORTS | SUPPORTS | SUPPORTS | Y | Y | Y | B |
| pp-005 | paraphrase | SUPPORTS | SUPPORTS | SUPPORTS | Y | Y | Y | - |
| pp-006 | paraphrase | SUPPORTS | PARTIAL | SUPPORTS | N | Y | Y | A |
| ps-001 | partial_support | PARTIAL | PARTIAL | PARTIAL | Y | Y | Y | B |
| ps-002 | partial_support | PARTIAL | SUPPORTS | SUPPORTS | N | N | Y | A,B |
| ps-003 | partial_support | PARTIAL | SUPPORTS | SUPPORTS | N | N | Y | B |
| ps-004 | partial_support | PARTIAL | PARTIAL | SUPPORTS | Y | N | Y | A |
| ps-005 | partial_support | PARTIAL | CONTRADICTS | SUPPORTS | N | N | Y | A,B |
| ps-006 | partial_support | PARTIAL | CONTRADICTS | SUPPORTS | N | N | Y | A,B |
| ec-001 | explicit_contradiction | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | N | - |
| ec-002 | explicit_contradiction | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | N | - |
| ec-003 | explicit_contradiction | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | N | - |
| ec-004 | explicit_contradiction | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | N | - |
| ec-005 | explicit_contradiction | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | N | - |
| ic-001 | implicit_contradiction | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | N | - |
| ic-002 | implicit_contradiction | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | N | - |
| ic-003 | implicit_contradiction | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | Y | A |
| ic-004 | implicit_contradiction | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | N | - |
| ic-005 | implicit_contradiction | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | N | - |
| os-001 | over_specificity | PARTIAL | CONTRADICTS | SUPPORTS | N | N | Y | - |
| os-002 | over_specificity | PARTIAL | SUPPORTS | SUPPORTS | N | N | Y | B |
| os-003 | over_specificity | PARTIAL | PARTIAL | SUPPORTS | Y | N | Y | A,B |
| os-004 | over_specificity | PARTIAL | SUPPORTS | SUPPORTS | N | N | Y | - |
| os-005 | over_specificity | PARTIAL | PARTIAL | SUPPORTS | Y | N | Y | A |
| n-001 | negation | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | N | - |
| n-002 | negation | PARTIAL | CONTRADICTS | PARTIAL | N | Y | Y | A |
| n-003 | negation | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | N | - |
| n-004 | negation | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | N | - |
| n-005 | negation | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | N | - |
| n-006 | negation | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | N | - |
| ws-001 | wrong_subject | UNRELATED | CONTRADICTS | CONTRADICTS | N | N | Y | A |
| ws-002 | wrong_subject | UNRELATED | UNRELATED | PARTIAL | Y | N | Y | A,B,D |
| ws-003 | wrong_subject | UNRELATED | UNRELATED | PARTIAL | Y | N | Y | A,B,D |
| ws-004 | wrong_subject | UNRELATED | CONTRADICTS | CONTRADICTS | N | N | Y | A,D |
| ws-005 | wrong_subject | UNRELATED | UNRELATED | CONTRADICTS | Y | N | Y | A,B |
| wc-001 | wrong_context | UNRELATED | CONTRADICTS | CONTRADICTS | N | N | Y | A |
| wc-002 | wrong_context | UNRELATED | CONTRADICTS | CONTRADICTS | N | N | Y | A |
| wc-003 | wrong_context | UNRELATED | CONTRADICTS | CONTRADICTS | N | N | Y | A,B,D |
| wc-004 | wrong_context | UNRELATED | CONTRADICTS | CONTRADICTS | N | N | N | - |
| wc-005 | wrong_context | PARTIAL | CONTRADICTS | CONTRADICTS | N | N | Y | A,D |
| a-001 | adversarial | PARTIAL | PARTIAL | SUPPORTS | Y | N | Y | A |
| a-002 | adversarial | PARTIAL | PARTIAL | SUPPORTS | Y | N | Y | A |
| a-003 | adversarial | PARTIAL | SUPPORTS | SUPPORTS | N | N | Y | - |
| a-004 | adversarial | PARTIAL | CONTRADICTS | CONTRADICTS | N | N | Y | A |
| a-005 | adversarial | CONTRADICTS | CONTRADICTS | CONTRADICTS | Y | Y | N | - |
| a-006 | adversarial | PARTIAL | CONTRADICTS | SUPPORTS | N | N | Y | A,B |

## Conclusion

Mode: `debate-on-disagreement`. The deliberation produced a **net negative** effect: 2 corrections vs 8 damage (net -6). Accuracy changed from 65.5% to 54.5% (-10.9%). The debate introduced more errors than it corrected. H0 is supported.
