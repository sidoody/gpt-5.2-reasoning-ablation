# Final Statistical Report

All outputs are derived deterministically from files under `results/` and `scores/`.

## Per-variant diagnosis accuracy

| Variant | N | Accuracy | 95% CI | Avg total tokens | Avg reasoning tokens | Avg latency (s) |
|---|---:|---:|---:|---:|---:|---:|
| none | 897 | 0.639 | [0.607, 0.670] | 613.61 | 0.00 | 2.608 |
| low | 897 | 0.664 | [0.633, 0.695] | 782.13 | 163.83 | 5.549 |
| medium | 897 | 0.673 | [0.642, 0.703] | 935.39 | 315.06 | 10.807 |
| high | 897 | 0.688 | [0.657, 0.717] | 1088.05 | 468.38 | 13.567 |

## Pairwise exact McNemar tests

| Comparison | N | a_correct_b_incorrect | a_incorrect_b_correct | Discordant total | Exact p-value |
|---|---:|---:|---:|---:|---:|
| none_vs_low | 897 | 48 | 71 | 119 | 0.04326826 |
| none_vs_medium | 897 | 44 | 75 | 119 | 0.00572686 |
| none_vs_high | 897 | 44 | 88 | 132 | 0.00016024 |
| low_vs_high | 897 | 36 | 57 | 93 | 0.03751423 |

## Cost/latency efficiency tradeoff

| Variant | Accuracy | Avg total tokens | Avg reasoning tokens | Avg latency (s) | Gain vs previous (pp) | Extra tokens vs previous | Extra latency vs previous (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| none | 0.639 | 613.61 | 0.00 | 2.608 | - | - | - |
| low | 0.664 | 782.13 | 163.83 | 5.549 | 2.5641 | 168.52 | 2.941 |
| medium | 0.673 | 935.39 | 315.06 | 10.807 | 0.8919 | 153.26 | 5.258 |
| high | 0.688 | 1088.05 | 468.38 | 13.567 | 1.4492 | 152.66 | 2.760 |
