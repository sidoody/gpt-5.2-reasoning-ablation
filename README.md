# GPT-5.2 Reasoning Effort Ablation

This repository asks one research question:

> Does increasing GPT-5.2 reasoning effort materially improve diagnosis accuracy, and is the gain worth the token/latency cost?

**Main finding (N=897 paired cases):** diagnosis accuracy rises from `0.639` (`none`) to `0.688` (`high`), but each step adds substantial token and latency cost. Pairwise exact McNemar tests show statistically significant gains for `none vs low`, `none vs medium`, `none vs high`, and `low vs high`.

**Benchmark caveat:** this is a case-report-heavy, rare-disease-skewed dataset. Treat this as a controlled ablation study, not a general-population clinical benchmark.

## Main Result

From `reports/summary_metrics.json` and `reports/pairwise_stats.json`:

| Variant | N | Accuracy | 95% CI | Avg total tokens | Avg latency (s) |
|---|---:|---:|---:|---:|---:|
| none | 897 | 0.639 | [0.607, 0.670] | 613.61 | 2.608 |
| low | 897 | 0.664 | [0.633, 0.695] | 782.13 | 5.549 |
| medium | 897 | 0.673 | [0.642, 0.703] | 935.39 | 10.807 |
| high | 897 | 0.688 | [0.657, 0.717] | 1088.05 | 13.567 |

Pairwise exact McNemar p-values:

- `none vs low`: `0.04326826` (discordant: 48 vs 71)
- `none vs medium`: `0.00572686` (discordant: 44 vs 75)
- `none vs high`: `0.00016024` (discordant: 44 vs 88)
- `low vs high`: `0.03751423` (discordant: 36 vs 57)

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
cp .env.example .env
# add OPENAI_API_KEY=...
```

Run a full study:

```bash
gpt52-ablation run --variants none low medium high
gpt52-ablation grade --variants none low medium high
gpt52-ablation report
```

The `report` command is the one-command publish step for analysis artifacts.

## Publishable Artifacts

`gpt52-ablation report` writes deterministic outputs under `reports/`:

- `summary_metrics.csv` and `summary_metrics.json`
- `pairwise_stats.csv` and `pairwise_stats.json`
- `cost_latency_tradeoffs.csv` and `cost_latency_tradeoffs.json`
- `final_report.md`
- `discordant_none_vs_high.json` (manual audit helper)

These files are designed to be quoted directly in README/blog/LinkedIn posts.

## Discordant Case Audit Helper

Export reviewable paired disagreements (default: `none` vs `high`):

```bash
gpt52-ablation export-discordant --a none --b high --limit 30
```

Each exported row includes:

- case ID
- gold diagnosis
- both predictions and correctness labels
- visible rationale summaries
- grader diagnosis/reasoning explanations

## Method Snapshot

- Evaluated model: `gpt-5.2`
- Variants: `none`, `low`, `medium`, `high`
- Grader model (fixed): `gpt-4.1`
- Dataset: `zou-lab/MedCaseReasoning` (`test` split)
- Scoring: diagnosis correctness (`0/1`) and reasoning alignment (`0-4`)
- Reported statistics:
  - per-variant diagnosis accuracy + 95% confidence interval
  - paired exact McNemar tests
  - token/latency tradeoff and incremental gain tables

## Limitations

- **Rare-disease skew:** `MedCaseReasoning` is not representative of everyday case mix.
- **Judge-model grading:** labels depend on GPT-4.1 grader behavior, even with a fixed rubric.
- **Visible-rationale scoring:** reasoning is graded only from model-visible rationale output, not hidden chain-of-thought.
- **`xhigh` exclusion:** `xhigh` exists in exploratory runs but is excluded from the main public 4-variant comparison due to coverage/cost profile.
