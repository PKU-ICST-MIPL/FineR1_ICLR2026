# Matched 2048-token Bird Evaluation

| Model | World | Split | n | Accuracy/TI | Semantic | Ref-box rate |
|---|---|---|---:|---:|---:|---:|
| Plain | Closed | Seen | 3483 | 70.17 | N/A | 0.000 |
| Plain | Closed | Unseen | 2311 | 55.69 | N/A | 0.000 |
| Plain | Open | Seen | 3483 | 29.66 | 64.58 | 0.000 |
| Plain | Open | Unseen | 2311 | 15.27 | 50.08 | 0.000 |
| Rigid | Closed | Seen | 3483 | 69.65 | N/A | 0.123 |
| Rigid | Closed | Unseen | 2311 | 61.06 | N/A | 0.108 |
| Rigid | Open | Seen | 3483 | 77.69 | 88.12 | 0.565 |
| Rigid | Open | Unseen | 2311 | 14.67 | 48.20 | 0.553 |
| Balanced | Closed | Seen | 3483 | 86.13 | N/A | 0.113 |
| Balanced | Closed | Unseen | 2311 | 80.61 | N/A | 0.152 |
| Balanced | Open | Seen | 3483 | 55.38 | 77.42 | 0.125 |
| Balanced | Open | Unseen | 2311 | 15.45 | 49.78 | 0.128 |

Configuration: `max_new_tokens=2048`, `min_pixels=1024`, `max_pixels=589824`.

The Balanced closed-world scores are strongly confounded by the released benchmark's fixed correct-option position. See `shuffled_closed_summary.md` and the main Stage1 v1 report.

