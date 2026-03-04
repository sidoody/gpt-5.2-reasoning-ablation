from __future__ import annotations

from pathlib import Path

from .io_utils import average, read_json, write_json
from .paths import result_path, score_path
from .schemas import GradeFile, RunFile
from .settings import SUPPORTED_REASONING_LEVELS, ModelVariant, StudySettings


def _load_run_file(settings: StudySettings, reasoning_level: str) -> RunFile | None:
    variant = ModelVariant(model=settings.model, reasoning_effort=reasoning_level)
    payload = read_json(result_path(settings, variant))
    return RunFile.model_validate(payload) if payload else None


def _load_grade_file(settings: StudySettings, reasoning_level: str) -> GradeFile | None:
    variant = ModelVariant(model=settings.model, reasoning_effort=reasoning_level)
    payload = read_json(score_path(settings, variant))
    return GradeFile.model_validate(payload) if payload else None


def summarize_runs(settings: StudySettings, write_path: str | None = None) -> list[dict]:
    rows: list[dict] = []
    for level in SUPPORTED_REASONING_LEVELS:
        run = _load_run_file(settings, level)
        grade = _load_grade_file(settings, level)
        if not run or not grade:
            continue

        shared_case_ids = sorted(set(run.cases) & set(grade.cases))
        diagnosis_scores = [grade.cases[case_id].diagnosis_correctness_score for case_id in shared_case_ids]
        reasoning_scores = [grade.cases[case_id].reasoning_alignment_score for case_id in shared_case_ids]
        reasoning_passes = [1 if grade.cases[case_id].reasoning_alignment_score >= 3 else 0 for case_id in shared_case_ids]
        latencies = [run.cases[case_id].latency_seconds for case_id in shared_case_ids]
        total_tokens = [float(run.cases[case_id].usage.get("total_tokens", 0)) for case_id in shared_case_ids]
        reasoning_tokens = [float(run.cases[case_id].usage.get("reasoning_tokens", 0)) for case_id in shared_case_ids]

        rows.append(
            {
                "variant_id": run.variant["id"],
                "reasoning_effort": level,
                "cases_scored": len(shared_case_ids),
                "diagnosis_accuracy": round(average(diagnosis_scores), 4),
                "mean_reasoning_alignment": round(average(reasoning_scores), 4),
                "reasoning_pass_rate": round(average(reasoning_passes), 4),
                "avg_latency_seconds": round(average(latencies), 3),
                "avg_total_tokens": round(average(total_tokens), 2),
                "avg_reasoning_tokens": round(average(reasoning_tokens), 2),
            }
        )

    if write_path:
        write_json(Path(write_path), rows)

    if not rows:
        print("No complete run+grade pairs found.")
        return rows

    print("\n=== Summary ===")
    for row in rows:
        print(
            f"{row['reasoning_effort']:>6} | cases={row['cases_scored']:>4} | "
            f"diag_acc={row['diagnosis_accuracy']:.3f} | "
            f"reasoning_mean={row['mean_reasoning_alignment']:.3f} | "
            f"reasoning_pass={row['reasoning_pass_rate']:.3f} | "
            f"avg_tokens={row['avg_total_tokens']:.1f} | "
            f"avg_reasoning_tokens={row['avg_reasoning_tokens']:.1f}"
        )
    return rows


def _mcnemar_counts(grades_a: GradeFile, grades_b: GradeFile) -> dict[str, int]:
    shared = sorted(set(grades_a.cases) & set(grades_b.cases))
    both_right = a_only = b_only = both_wrong = 0
    for case_id in shared:
        a_correct = grades_a.cases[case_id].diagnosis_correctness_score == 1
        b_correct = grades_b.cases[case_id].diagnosis_correctness_score == 1
        if a_correct and b_correct:
            both_right += 1
        elif a_correct and not b_correct:
            a_only += 1
        elif not a_correct and b_correct:
            b_only += 1
        else:
            both_wrong += 1
    return {
        "shared": len(shared),
        "both_right": both_right,
        "a_only": a_only,
        "b_only": b_only,
        "both_wrong": both_wrong,
    }


def _mcnemar_statistic(a_only: int, b_only: int) -> float:
    discordant = a_only + b_only
    if discordant == 0:
        return 0.0
    return ((abs(a_only - b_only) - 1) ** 2) / discordant


def analyze_pairs(settings: StudySettings, write_path: str | None = None) -> list[dict]:
    levels = list(SUPPORTED_REASONING_LEVELS)
    comparisons: list[dict] = []

    for lower, higher in zip(levels[:-1], levels[1:]):
        lower_run = _load_run_file(settings, lower)
        higher_run = _load_run_file(settings, higher)
        lower_grade = _load_grade_file(settings, lower)
        higher_grade = _load_grade_file(settings, higher)
        if not all([lower_run, higher_run, lower_grade, higher_grade]):
            continue

        counts = _mcnemar_counts(lower_grade, higher_grade)
        shared_case_ids = sorted(set(lower_grade.cases) & set(higher_grade.cases))
        lower_reasoning = [lower_grade.cases[case_id].reasoning_alignment_score for case_id in shared_case_ids]
        higher_reasoning = [higher_grade.cases[case_id].reasoning_alignment_score for case_id in shared_case_ids]
        reasoning_delta = average(lower_reasoning) - average(higher_reasoning)

        examples = []
        for case_id in shared_case_ids:
            lower_correct = lower_grade.cases[case_id].diagnosis_correctness_score == 1
            higher_correct = higher_grade.cases[case_id].diagnosis_correctness_score == 1
            if lower_correct and not higher_correct:
                examples.append(
                    {
                        "case_id": case_id,
                        "lower_prediction": lower_run.cases[case_id].diagnosis,
                        "higher_prediction": higher_run.cases[case_id].diagnosis,
                        "higher_reasoning_explanation": higher_grade.cases[case_id].reasoning_explanation,
                    }
                )
            if len(examples) >= 10:
                break

        comparisons.append(
            {
                "lower_effort": lower,
                "higher_effort": higher,
                "cases_shared": counts["shared"],
                "lower_only_correct": counts["a_only"],
                "higher_only_correct": counts["b_only"],
                "mcnemar_chi_square_cc": round(_mcnemar_statistic(counts["a_only"], counts["b_only"]), 4),
                "mean_reasoning_alignment_delta": round(reasoning_delta, 4),
                "lower_beats_higher_examples": examples,
            }
        )

    if write_path:
        write_json(Path(write_path), comparisons)

    if not comparisons:
        print("No pairwise comparisons available yet.")
        return comparisons

    print("\n=== Pairwise analysis ===")
    for item in comparisons:
        print(
            f"{item['lower_effort']} -> {item['higher_effort']} | shared={item['cases_shared']} | "
            f"lower_only_correct={item['lower_only_correct']} | higher_only_correct={item['higher_only_correct']} | "
            f"mcnemar_cc={item['mcnemar_chi_square_cc']:.3f} | "
            f"reasoning_delta={item['mean_reasoning_alignment_delta']:.3f}"
        )
    return comparisons


def analyze_overthinking(settings: StudySettings, write_path: str | None = None) -> list[dict]:
    """Deprecated alias for analyze_pairs."""
    return analyze_pairs(settings, write_path=write_path)
