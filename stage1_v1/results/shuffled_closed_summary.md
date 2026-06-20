# Deterministically Shuffled Closed-World Control

Seed: `20260620`.

| Model | Seen accuracy | Unseen accuracy | Status |
|---|---:|---:|---|
| Plain | 65.69 | 51.49 | Complete |
| Rigid | **65.98** | **55.95** | Complete |
| Balanced | 54.49 | Pending at v1 report time | Running |

The released Bird files place the correct answer at index 0 in 3483/3483 seen and 2311/2311 unseen examples. The shuffled control changes only option order and preserves the released prompt, images, answer parser, generation length, and processor budget.

Paired Plain vs Rigid:

| Split | Plain-only correct | Rigid-only correct | Continuity-corrected McNemar p |
|---|---:|---:|---:|
| Seen | 335 | 345 | 0.730 |
| Unseen | 196 | 299 | `4.55e-6` |

