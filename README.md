# GPT-5.2 Reasoning Effort Ablation

A reproducible evaluation of how **GPT-5.2** diagnosis performance changes as reasoning effort increases from `none` to `low`, `medium`, and `high`.

This repository is structured as a complete study package:

- **committed model outputs** in `results/`
- **committed grader outputs** in `scores/`
- **deterministic analysis artifacts** in `reports/`
- **CLI workflows** for running, grading, and regenerating reports

The goal is simple: measure whether additional reasoning effort buys enough diagnostic accuracy to justify the added **latency** and **token cost**.

---

## Why this repo exists

Reasoning settings are increasingly exposed as a deployment knob, but the practical question is not just *whether* more reasoning helps. It is:

> **How much additional accuracy do you buy, and what do you pay for it in latency and tokens?**

This repo answers that question on a paired clinical benchmark using fixed grading, committed outputs, and reproducible reporting.

---

## Top-line results

On **897 paired benchmark cases**, diagnosis accuracy increased from **0.639** at `none` to **0.688** at `high`, while average latency increased from **2.608s** to **13.567s** and average total tokens increased from **613.61** to **1088.05**.

### Per-variant diagnosis accuracy

| Variant | N | Accuracy | 95% CI | Avg total tokens | Avg reasoning tokens | Avg latency (s) |
|---|---:|---:|---:|---:|---:|---:|
| none | 897 | 0.639 | [0.607, 0.670] | 613.61 | 0.00 | 2.608 |
| low | 897 | 0.664 | [0.633, 0.695] | 782.13 | 163.83 | 5.549 |
| medium | 897 | 0.673 | [0.642, 0.703] | 935.39 | 315.06 | 10.807 |
| high | 897 | 0.688 | [0.657, 0.717] | 1088.05 | 468.38 | 13.567 |

### All-pairs exact McNemar tests

| Comparison | N | Accuracy A | Accuracy B | \|Delta\| | A-only correct | B-only correct | Discordant total | Exact p-value | Holm-adjusted p-value |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| none_vs_low | 897 | 0.639 | 0.664 | 0.026 | 48 | 71 | 119 | 0.04326826 | 0.15005693 |
| none_vs_medium | 897 | 0.639 | 0.673 | 0.035 | 44 | 75 | 119 | 0.00572686 | 0.02863431 |
| none_vs_high | 897 | 0.639 | 0.688 | 0.049 | 44 | 88 | 132 | 0.00016024 | 0.00096141 |
| low_vs_medium | 897 | 0.664 | 0.673 | 0.009 | 41 | 49 | 90 | 0.46079247 | 0.46079247 |
| low_vs_high | 897 | 0.664 | 0.688 | 0.023 | 36 | 57 | 93 | 0.03751423 | 0.15005693 |
| medium_vs_high | 897 | 0.673 | 0.688 | 0.014 | 36 | 49 | 85 | 0.19276044 | 0.38552087 |

### Efficiency frontier

| Variant | Accuracy | Avg total tokens | Avg reasoning tokens | Avg latency (s) |
|---|---:|---:|---:|---:|
| none | 0.639 | 613.61 | 0.00 | 2.608 |
| low | 0.664 | 782.13 | 163.83 | 5.549 |
| medium | 0.673 | 935.39 | 315.06 | 10.807 |
| high | 0.688 | 1088.05 | 468.38 | 13.567 |

A few deployment-relevant takeaways from these results:

- `none -> low` buys **+2.56 percentage points** of accuracy with modest additional cost.
- `none -> high` buys the largest gain: **+4.91 points**, but at substantially higher latency and token usage.
- `low -> high` still improves accuracy (**+2.34 points**), but at a much steeper cost than `none -> low`.
- After Holm correction across the full pair set, the strongest evidence remains for **`none -> high`** and **`none -> medium`** on this benchmark.

---

## Study question

How do GPT-5.2 reasoning-effort settings trade off **diagnosis accuracy** against **latency** and **token cost** on the benchmark cases used in this repository?

---

## Method overview

- **Model under evaluation:** `gpt-5.2`
- **Variants:** `none`, `low`, `medium`, `high`
- **Dataset:** `zou-lab/MedCaseReasoning` (`test`)
- **Grader model:** `gpt-4.1` (held fixed across all variants)
- **Primary outcome:** diagnosis correctness (`0/1`)
- **Comparison design:** paired-case comparisons using shared case IDs only

Reported statistics include:

- per-variant accuracy with **95% Wilson intervals**
- **all-pairs exact McNemar tests**
- **Holm-adjusted p-values** across the full pair set
- deployment-oriented tradeoff measures:
  - additional correct cases per 1,000
  - extra tokens
  - extra latency
  - cost-efficiency ratios

---

## What was held constant

This repository is designed to isolate the effect of **reasoning effort** while keeping the rest of the evaluation stack fixed.

Held constant across variants:

- evaluated model family (`gpt-5.2`)
- benchmark dataset and split
- grading model (`gpt-4.1`)
- grading rubric
- reporting pipeline inputs (`results/` and `scores/`)

This means the analysis is about one controlled question:

> **What changes when reasoning effort changes?**

---

## Immutable vs. derived data

The repo makes a strict distinction between **study outputs** and **derived reports**.

### Immutable source-of-truth artifacts
- `results/` - raw model outputs
- `scores/` - grader outputs

### Derived, reproducible artifacts
- `reports/` - summaries, pairwise statistics, plots, and markdown reports regenerated from `results/` and `scores/`

That separation matters: the experiment outputs are committed and stable, while the analysis layer can be regenerated, extended, or audited without rerunning inference.

---

## Reproducible reporting

Regenerate all report artifacts from the committed study outputs with:

```bash
pip install -e .
gpt52-ablation report
```

The report command:

- validates committed inputs
- rebuilds `reports/` from scratch
- does **not** rerun model inference
- does **not** rerun grading

---

## Generated report artifacts

Running `gpt52-ablation report` writes:

- `reports/variant_summary.json`
- `reports/variant_summary.csv`
- `reports/pairwise_matrix.json`
- `reports/pairwise_matrix.csv`
- `reports/deployment_views.json`
- `reports/deployment_views.csv`
- `reports/efficiency_frontier.json`
- `reports/efficiency_frontier.csv`
- `reports/validation_summary.json`
- `reports/pairwise_mcnemar_p_values.svg`
- `reports/final_report.md`
- `reports/discordant_case_exports/*.json`

If you want the fully rendered results first, start with:

- `reports/final_report.md`
- `reports/pairwise_matrix.csv`
- `reports/deployment_views.csv`

---

## Common workflows

### 1. Run inference

```bash
gpt52-ablation run --variants none low medium high
```

### 2. Grade saved runs

```bash
gpt52-ablation grade --variants none low medium high
```

### 3. Regenerate reports from committed outputs

```bash
gpt52-ablation report
```

### 4. Export discordant cases for qualitative review

```bash
gpt52-ablation export-discordant --a none --b high --limit 30
```

### 5. Inspect pairwise analysis directly

```bash
gpt52-ablation analyze-pairs
```

---

## What questions this repo supports

This repository is most useful for questions like:

- Should reasoning be enabled by default?
- How much additional accuracy does `low` buy over `none`?
- If some reasoning is already enabled, is `high` worth the extra latency and token cost?
- What is the maximum-accuracy setting on this benchmark?
- Which settings lie on the practical frontier of accuracy vs. latency vs. tokens?

In other words, this repo is intended to support **deployment-style interpretation**, not just adjacent-step significance testing.

---

## Interpretation scope

This repository supports conclusions about:

- measured diagnosis-accuracy differences on this benchmark
- paired-case comparisons between observed variants
- token and latency tradeoffs for these specific runs
- how different reasoning-effort settings move along the benchmark’s efficiency frontier

This repository does **not** establish:

- real-world prevalence estimates
- standalone clinical safety claims
- replacement of clinician judgment
- general clinical deployment readiness
- broad generalization beyond this benchmark without further validation

---

## Benchmark caveats

This benchmark should be interpreted with care.

### Dataset skew
`MedCaseReasoning` is not a general-population clinical dataset. It is skewed toward more complex diagnostic cases and should not be treated as representative of routine case mix.

### Judge-model dependence
Grading is held fixed with `gpt-4.1`, which improves consistency, but it also means the scoring pipeline depends on a model-based evaluator.

### Benchmark conclusions are not deployment conclusions
A statistically significant paired benchmark improvement is not the same as a clinical safety or workflow-readiness claim. Real deployment decisions would require additional datasets, workflow testing, monitoring, and governance.

---

## Repository structure

- `src/gpt_5_2_reasoning_ablation/runner.py` - inference pipeline
- `src/gpt_5_2_reasoning_ablation/grading.py` - grading pipeline
- `src/gpt_5_2_reasoning_ablation/reporting.py` - deterministic report generation
- `src/gpt_5_2_reasoning_ablation/analysis.py` - pairwise statistical analysis
- `results/` - committed raw model outputs
- `scores/` - committed grader outputs
- `reports/` - generated analysis and reporting artifacts
- `tests/` - automated tests for CLI, schemas, analysis, reporting, and report regeneration

---

## Why the analysis is pairwise

Earlier versions of this project emphasized adjacent comparisons such as `none -> low` and `low -> medium`. That framing is useful, but incomplete.

The current reporting layer evaluates **all unique pairwise comparisons** across the observed variants. This gives a better view of the real decision space, including comparisons such as:

- `none -> high`
- `none -> medium`
- `low -> high`

For practical deployment decisions, those comparisons are often more informative than adjacent-only step-ups.

---

## Reproducing the full study

If you want to rerun inference and grading yourself rather than relying on the committed outputs:

```bash
pip install -e .

gpt52-ablation run --variants none low medium high
gpt52-ablation grade --variants none low medium high
gpt52-ablation report
```

For most users, rerunning the full benchmark is unnecessary. The committed `results/` and `scores/` already support deterministic report regeneration.

---

## Summary

This repository provides a reproducible answer to a concrete question:

> **How much diagnostic accuracy does additional GPT-5.2 reasoning effort buy, and what does it cost?**

On this benchmark, more reasoning does improve accuracy, but the gains are not uniform across comparisons, and they come with substantial latency and token tradeoffs. The repo is designed so those tradeoffs can be inspected directly from committed outputs rather than inferred from prose alone.

---

## License

MIT
