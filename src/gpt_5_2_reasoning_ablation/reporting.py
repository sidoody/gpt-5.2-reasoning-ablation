from __future__ import annotations

import csv
import math
from pathlib import Path

from .io_utils import average, write_json
from .paths import reports_dir
from .schemas import GradeFile, RunFile
from .settings import DEFAULT_REASONING_LEVELS, ModelVariant, StudySettings
from .analysis import _load_grade_file, _load_run_file

PREFERRED_VARIANTS = ("none", "low", "medium", "high")
PREFERRED_PAIRS = (
    ("none", "low"),
    ("low", "medium"),
    ("medium", "high"),
)
WILSON_Z_95 = 1.959963984540054


def _variant_rows(settings: StudySettings) -> list[dict]:
    rows: list[dict] = []
    for level in PREFERRED_VARIANTS:
        run = _load_run_file(settings, level)
        grade = _load_grade_file(settings, level)
        if not run or not grade:
            continue

        shared_case_ids = sorted(set(run.cases) & set(grade.cases))
        diagnosis_scores = [grade.cases[case_id].diagnosis_correctness_score for case_id in shared_case_ids]
        latencies = [run.cases[case_id].latency_seconds for case_id in shared_case_ids]
        total_tokens = [float(run.cases[case_id].usage.get("total_tokens", 0)) for case_id in shared_case_ids]
        reasoning_tokens = [float(run.cases[case_id].usage.get("reasoning_tokens", 0)) for case_id in shared_case_ids]

        n = len(shared_case_ids)
        correct = int(sum(diagnosis_scores))
        accuracy = average([float(score) for score in diagnosis_scores])
        ci_low, ci_high = _wilson_interval(correct, n)
        rows.append(
            {
                "reasoning_effort": level,
                "variant_id": run.variant["id"],
                "n": n,
                "correct": correct,
                "accuracy": round(accuracy, 6),
                "accuracy_ci95_low": round(ci_low, 6),
                "accuracy_ci95_high": round(ci_high, 6),
                "avg_total_tokens": round(average(total_tokens), 2),
                "avg_reasoning_tokens": round(average(reasoning_tokens), 2),
                "avg_latency_seconds": round(average(latencies), 3),
            }
        )
    return rows


def _wilson_interval(successes: int, total: int, z: float = WILSON_Z_95) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    p_hat = successes / total
    z2 = z * z
    denom = 1 + z2 / total
    center = (p_hat + z2 / (2 * total)) / denom
    margin = (z / denom) * math.sqrt((p_hat * (1 - p_hat) / total) + (z2 / (4 * total * total)))
    low = max(0.0, center - margin)
    high = min(1.0, center + margin)
    return low, high


def _mcnemar_counts(grades_a: GradeFile, grades_b: GradeFile) -> dict[str, int]:
    shared = sorted(set(grades_a.cases) & set(grades_b.cases))
    a_correct_b_incorrect = 0
    a_incorrect_b_correct = 0

    for case_id in shared:
        a_correct = grades_a.cases[case_id].diagnosis_correctness_score == 1
        b_correct = grades_b.cases[case_id].diagnosis_correctness_score == 1
        if a_correct and not b_correct:
            a_correct_b_incorrect += 1
        elif not a_correct and b_correct:
            a_incorrect_b_correct += 1

    return {
        "n": len(shared),
        "a_correct_b_incorrect": a_correct_b_incorrect,
        "a_incorrect_b_correct": a_incorrect_b_correct,
        "discordant_total": a_correct_b_incorrect + a_incorrect_b_correct,
    }


def _mcnemar_exact_p_value(a_correct_b_incorrect: int, a_incorrect_b_correct: int) -> float:
    discordant = a_correct_b_incorrect + a_incorrect_b_correct
    if discordant == 0:
        return 1.0
    min_side = min(a_correct_b_incorrect, a_incorrect_b_correct)
    tail_prob = sum(math.comb(discordant, i) for i in range(min_side + 1)) / (2**discordant)
    return min(1.0, 2.0 * tail_prob)


def _pairwise_rows(settings: StudySettings) -> list[dict]:
    rows: list[dict] = []
    for a_level, b_level in PREFERRED_PAIRS:
        grade_a = _load_grade_file(settings, a_level)
        grade_b = _load_grade_file(settings, b_level)
        if not grade_a or not grade_b:
            continue
        counts = _mcnemar_counts(grade_a, grade_b)
        rows.append(
            {
                "comparison": f"{a_level}_vs_{b_level}",
                "a_level": a_level,
                "b_level": b_level,
                "n": counts["n"],
                "a_correct_b_incorrect": counts["a_correct_b_incorrect"],
                "a_incorrect_b_correct": counts["a_incorrect_b_correct"],
                "discordant_total": counts["discordant_total"],
                "mcnemar_exact_p_value": round(
                    _mcnemar_exact_p_value(
                        counts["a_correct_b_incorrect"],
                        counts["a_incorrect_b_correct"],
                    ),
                    8,
                ),
            }
        )
    return rows


def _cost_tradeoff_rows(variant_rows: list[dict]) -> list[dict]:
    rows: list[dict] = []
    previous: dict | None = None
    for row in variant_rows:
        current = {
            "reasoning_effort": row["reasoning_effort"],
            "n": row["n"],
            "accuracy": row["accuracy"],
            "avg_total_tokens": row["avg_total_tokens"],
            "avg_reasoning_tokens": row["avg_reasoning_tokens"],
            "avg_latency_seconds": row["avg_latency_seconds"],
            "accuracy_gain_vs_previous_pp": None,
            "extra_total_tokens_vs_previous": None,
            "extra_reasoning_tokens_vs_previous": None,
            "extra_latency_seconds_vs_previous": None,
            "extra_tokens_per_1pp_gain": None,
            "extra_latency_seconds_per_1pp_gain": None,
        }
        if previous:
            gain_pp = (row["accuracy"] - previous["accuracy"]) * 100
            extra_total = row["avg_total_tokens"] - previous["avg_total_tokens"]
            extra_reasoning = row["avg_reasoning_tokens"] - previous["avg_reasoning_tokens"]
            extra_latency = row["avg_latency_seconds"] - previous["avg_latency_seconds"]
            current["accuracy_gain_vs_previous_pp"] = round(gain_pp, 4)
            current["extra_total_tokens_vs_previous"] = round(extra_total, 2)
            current["extra_reasoning_tokens_vs_previous"] = round(extra_reasoning, 2)
            current["extra_latency_seconds_vs_previous"] = round(extra_latency, 3)
            if gain_pp > 0:
                current["extra_tokens_per_1pp_gain"] = round(extra_total / gain_pp, 2)
                current["extra_latency_seconds_per_1pp_gain"] = round(extra_latency / gain_pp, 4)
        rows.append(current)
        previous = row
    return rows


def _pairwise_p_value_chart_svg(pairwise_rows: list[dict]) -> str:
    width = 860
    height = 360
    margin_left = 80
    margin_right = 40
    margin_top = 40
    margin_bottom = 90
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    if not pairwise_rows:
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
            '<rect width="100%" height="100%" fill="#ffffff"/>'
            '<text x="50%" y="50%" text-anchor="middle" fill="#4b5563" font-size="16" font-family="Arial, sans-serif">'
            "No pairwise rows available."
            "</text>"
            "</svg>"
        )

    bars: list[tuple[str, float, float]] = []
    for row in pairwise_rows:
        label = f"{row['a_level']}->{row['b_level']}"
        p_value = max(float(row["mcnemar_exact_p_value"]), 1e-16)
        neg_log10 = -math.log10(p_value)
        bars.append((label, p_value, neg_log10))

    y_max = max(1.0, max(neg_log10 for _, _, neg_log10 in bars) * 1.15)
    zero_y = margin_top + plot_height
    threshold_value = -math.log10(0.05)
    threshold_y = margin_top + plot_height - (threshold_value / y_max) * plot_height
    bar_width = plot_width / max(len(bars), 1) * 0.55

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="50%" y="24" text-anchor="middle" fill="#111827" font-size="18" font-family="Arial, sans-serif">Pairwise McNemar exact p-values</text>',
        f'<line x1="{margin_left}" y1="{zero_y}" x2="{width - margin_right}" y2="{zero_y}" stroke="#111827" stroke-width="1.5"/>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{zero_y}" stroke="#111827" stroke-width="1.5"/>',
        f'<line x1="{margin_left}" y1="{threshold_y}" x2="{width - margin_right}" y2="{threshold_y}" stroke="#ef4444" stroke-width="1.2" stroke-dasharray="6 5"/>',
        f'<text x="{width - margin_right - 8}" y="{threshold_y - 6}" text-anchor="end" fill="#ef4444" font-size="11" font-family="Arial, sans-serif">p=0.05</text>',
        f'<text x="{margin_left - 54}" y="{margin_top - 8}" fill="#374151" font-size="11" font-family="Arial, sans-serif">-log10(p)</text>',
    ]

    tick_steps = 4
    for tick_index in range(tick_steps + 1):
        tick_value = (tick_index / tick_steps) * y_max
        y = margin_top + plot_height - (tick_value / y_max) * plot_height
        tick_label = f"{tick_value:.2f}".rstrip("0").rstrip(".")
        parts.append(
            f'<line x1="{margin_left - 4}" y1="{y}" x2="{margin_left}" y2="{y}" stroke="#111827" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{margin_left - 10}" y="{y + 4}" text-anchor="end" fill="#374151" font-size="10" font-family="Arial, sans-serif">{tick_label}</text>'
        )

    slot_width = plot_width / max(len(bars), 1)
    for index, (label, p_value, neg_log10) in enumerate(bars):
        x_center = margin_left + slot_width * (index + 0.5)
        bar_height = (neg_log10 / y_max) * plot_height
        bar_x = x_center - (bar_width / 2)
        bar_y = zero_y - bar_height
        parts.append(
            f'<rect x="{bar_x:.2f}" y="{bar_y:.2f}" width="{bar_width:.2f}" height="{bar_height:.2f}" fill="#3b82f6"/>'
        )
        parts.append(
            f'<text x="{x_center:.2f}" y="{zero_y + 20}" text-anchor="middle" fill="#111827" font-size="11" font-family="Arial, sans-serif">{label}</text>'
        )
        parts.append(
            f'<text x="{x_center:.2f}" y="{max(bar_y - 6, margin_top + 12):.2f}" text-anchor="middle" fill="#1f2937" font-size="10" font-family="Arial, sans-serif">p={p_value:.8f}</text>'
        )

    parts.append("</svg>")
    return "".join(parts)


def _write_pairwise_p_value_chart(path: Path, pairwise_rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_pairwise_p_value_chart_svg(pairwise_rows))


def _visible_rationale(run: RunFile, case_id: str) -> str:
    case = run.cases[case_id]
    bullets = [item.strip() for item in case.rationale_summary if item and item.strip()]
    if case.api_reasoning_summary:
        bullets.append(f"API summary: {case.api_reasoning_summary.strip()}")
    return " | ".join(bullets)


def export_discordant_cases(
    settings: StudySettings,
    a_level: str = "none",
    b_level: str = "high",
    limit: int = 30,
    write_path: str | None = None,
) -> list[dict]:
    run_a = _load_run_file(settings, a_level)
    run_b = _load_run_file(settings, b_level)
    grade_a = _load_grade_file(settings, a_level)
    grade_b = _load_grade_file(settings, b_level)
    if not all([run_a, run_b, grade_a, grade_b]):
        return []

    shared_case_ids = sorted(set(run_a.cases) & set(run_b.cases) & set(grade_a.cases) & set(grade_b.cases))
    discordant: list[dict] = []

    for case_id in shared_case_ids:
        a_outcome = grade_a.cases[case_id]
        b_outcome = grade_b.cases[case_id]
        a_correct = a_outcome.diagnosis_correctness_score == 1
        b_correct = b_outcome.diagnosis_correctness_score == 1
        if a_correct == b_correct:
            continue
        discordant.append(
            {
                "case_id": case_id,
                "gold_diagnosis": a_outcome.ground_truth_diagnosis,
                "comparison": f"{a_level}_vs_{b_level}",
                "a_level": a_level,
                "a_prediction": run_a.cases[case_id].diagnosis,
                "a_correctness_label": a_outcome.diagnosis_correctness_label,
                "a_visible_rationale_summary": _visible_rationale(run_a, case_id),
                "a_grader_diagnosis_explanation": a_outcome.diagnosis_explanation,
                "a_grader_reasoning_explanation": a_outcome.reasoning_explanation,
                "b_level": b_level,
                "b_prediction": run_b.cases[case_id].diagnosis,
                "b_correctness_label": b_outcome.diagnosis_correctness_label,
                "b_visible_rationale_summary": _visible_rationale(run_b, case_id),
                "b_grader_diagnosis_explanation": b_outcome.diagnosis_explanation,
                "b_grader_reasoning_explanation": b_outcome.reasoning_explanation,
            }
        )
        if len(discordant) >= limit:
            break

    destination = Path(write_path) if write_path else reports_dir(settings) / f"discordant_{a_level}_vs_{b_level}.json"
    write_json(destination, discordant)
    return discordant


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _format_metric(x: float) -> str:
    return f"{x:.3f}"


def _format_ci(low: float, high: float) -> str:
    return f"[{low:.3f}, {high:.3f}]"


def _markdown_report(variant_rows: list[dict], pairwise_rows: list[dict], cost_rows: list[dict]) -> str:
    lines = [
        "# Final Statistical Report",
        "",
        "All outputs are derived deterministically from files under `results/` and `scores/`.",
        "",
        "## Per-variant diagnosis accuracy",
        "",
        "| Variant | N | Accuracy | 95% CI | Avg total tokens | Avg reasoning tokens | Avg latency (s) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in variant_rows:
        lines.append(
            "| "
            f"{row['reasoning_effort']} | "
            f"{row['n']} | "
            f"{_format_metric(row['accuracy'])} | "
            f"{_format_ci(row['accuracy_ci95_low'], row['accuracy_ci95_high'])} | "
            f"{row['avg_total_tokens']:.2f} | "
            f"{row['avg_reasoning_tokens']:.2f} | "
            f"{row['avg_latency_seconds']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Pairwise exact McNemar tests",
            "",
            "| Comparison | N | a_correct_b_incorrect | a_incorrect_b_correct | Discordant total | Exact p-value |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in pairwise_rows:
        lines.append(
            "| "
            f"{row['comparison']} | "
            f"{row['n']} | "
            f"{row['a_correct_b_incorrect']} | "
            f"{row['a_incorrect_b_correct']} | "
            f"{row['discordant_total']} | "
            f"{row['mcnemar_exact_p_value']:.8f} |"
        )

    lines.extend(
        [
            "",
            "## Cost/latency efficiency tradeoff",
            "",
            "| Variant | Accuracy | Avg total tokens | Avg reasoning tokens | Avg latency (s) | Gain vs previous (pp) | Extra tokens vs previous | Extra latency vs previous (s) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in cost_rows:
        gain = "-" if row["accuracy_gain_vs_previous_pp"] is None else f"{row['accuracy_gain_vs_previous_pp']:.4f}"
        extra_tokens = "-" if row["extra_total_tokens_vs_previous"] is None else f"{row['extra_total_tokens_vs_previous']:.2f}"
        extra_latency = "-" if row["extra_latency_seconds_vs_previous"] is None else f"{row['extra_latency_seconds_vs_previous']:.3f}"
        lines.append(
            "| "
            f"{row['reasoning_effort']} | "
            f"{_format_metric(row['accuracy'])} | "
            f"{row['avg_total_tokens']:.2f} | "
            f"{row['avg_reasoning_tokens']:.2f} | "
            f"{row['avg_latency_seconds']:.3f} | "
            f"{gain} | "
            f"{extra_tokens} | "
            f"{extra_latency} |"
        )
    lines.append("")
    return "\n".join(lines)


def generate_final_artifacts(settings: StudySettings, discordant_limit: int = 30) -> dict[str, Path]:
    variant_rows = _variant_rows(settings)
    pairwise_rows = _pairwise_rows(settings)
    cost_rows = _cost_tradeoff_rows(variant_rows)

    base_dir = reports_dir(settings)
    summary_json = base_dir / "summary_metrics.json"
    summary_csv = base_dir / "summary_metrics.csv"
    pairwise_json = base_dir / "pairwise_stats.json"
    pairwise_csv = base_dir / "pairwise_stats.csv"
    cost_json = base_dir / "cost_latency_tradeoffs.json"
    cost_csv = base_dir / "cost_latency_tradeoffs.csv"
    pairwise_p_chart_svg = base_dir / "pairwise_mcnemar_p_values.svg"
    report_md = base_dir / "final_report.md"

    write_json(summary_json, variant_rows)
    write_json(pairwise_json, pairwise_rows)
    write_json(cost_json, cost_rows)
    _write_csv(summary_csv, variant_rows)
    _write_csv(pairwise_csv, pairwise_rows)
    _write_csv(cost_csv, cost_rows)
    _write_pairwise_p_value_chart(pairwise_p_chart_svg, pairwise_rows)
    report_md.write_text(_markdown_report(variant_rows, pairwise_rows, cost_rows))

    discordant_none_high = export_discordant_cases(
        settings,
        a_level="none",
        b_level="high",
        limit=discordant_limit,
        write_path=str(base_dir / "discordant_none_vs_high.json"),
    )

    print("\n=== Final report artifacts ===")
    print(f"summary metrics: {summary_json}")
    print(f"pairwise stats: {pairwise_json}")
    print(f"cost/latency tradeoffs: {cost_json}")
    print(f"pairwise p-value chart: {pairwise_p_chart_svg}")
    print(f"markdown report: {report_md}")
    print(f"discordant none_vs_high examples: {len(discordant_none_high)}")

    if variant_rows:
        print("\n=== Main result (copyable) ===")
        for row in variant_rows:
            print(
                f"{row['reasoning_effort']:>6} | N={row['n']:>4} | acc={row['accuracy']:.3f} "
                f"(95% CI {_format_ci(row['accuracy_ci95_low'], row['accuracy_ci95_high'])}) | "
                f"tokens={row['avg_total_tokens']:.1f} | latency={row['avg_latency_seconds']:.3f}s"
            )

    return {
        "summary_json": summary_json,
        "summary_csv": summary_csv,
        "pairwise_json": pairwise_json,
        "pairwise_csv": pairwise_csv,
        "cost_json": cost_json,
        "cost_csv": cost_csv,
        "pairwise_p_values_chart_svg": pairwise_p_chart_svg,
        "report_md": report_md,
        "discordant_none_high_json": base_dir / "discordant_none_vs_high.json",
    }
