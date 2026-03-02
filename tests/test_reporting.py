from __future__ import annotations

from pathlib import Path

from gpt_5_2_reasoning_ablation.io_utils import write_json
from gpt_5_2_reasoning_ablation.reporting import (
    _mcnemar_exact_p_value,
    _pairwise_rows,
    _variant_rows,
    _wilson_interval,
    export_discordant_cases,
)
from gpt_5_2_reasoning_ablation.settings import StudySettings


def _seed_files(settings: StudySettings) -> None:
    run_none = {
        "study_name": "gpt-5.2-reasoning-ablation",
        "dataset": {"name": "x", "split": "y"},
        "variant": {"id": "gpt-5.2__reasoning-none", "model": "gpt-5.2", "reasoning_effort": "none"},
        "run_settings": {},
        "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T00:00:00+00:00",
        "cases": {
            "C1": {
                "case_id": "C1",
                "diagnosis": "A",
                "rationale_summary": ["r1"],
                "raw_output_text": "{}",
                "parsed_output": {"diagnosis": "A", "rationale_summary": ["r1"]},
                "api_reasoning_summary": None,
                "latency_seconds": 1.0,
                "usage": {"total_tokens": 100, "reasoning_tokens": 10},
                "timestamp": "2026-01-01T00:00:00+00:00",
            },
            "C2": {
                "case_id": "C2",
                "diagnosis": "B",
                "rationale_summary": ["r2"],
                "raw_output_text": "{}",
                "parsed_output": {"diagnosis": "B", "rationale_summary": ["r2"]},
                "api_reasoning_summary": None,
                "latency_seconds": 2.0,
                "usage": {"total_tokens": 110, "reasoning_tokens": 11},
                "timestamp": "2026-01-01T00:00:00+00:00",
            },
        },
    }
    run_high = {
        "study_name": "gpt-5.2-reasoning-ablation",
        "dataset": {"name": "x", "split": "y"},
        "variant": {"id": "gpt-5.2__reasoning-high", "model": "gpt-5.2", "reasoning_effort": "high"},
        "run_settings": {},
        "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T00:00:00+00:00",
        "cases": {
            "C1": {
                "case_id": "C1",
                "diagnosis": "A2",
                "rationale_summary": ["h1"],
                "raw_output_text": "{}",
                "parsed_output": {"diagnosis": "A2", "rationale_summary": ["h1"]},
                "api_reasoning_summary": "extra",
                "latency_seconds": 3.0,
                "usage": {"total_tokens": 210, "reasoning_tokens": 90},
                "timestamp": "2026-01-01T00:00:00+00:00",
            },
            "C2": {
                "case_id": "C2",
                "diagnosis": "B",
                "rationale_summary": ["h2"],
                "raw_output_text": "{}",
                "parsed_output": {"diagnosis": "B", "rationale_summary": ["h2"]},
                "api_reasoning_summary": None,
                "latency_seconds": 4.0,
                "usage": {"total_tokens": 220, "reasoning_tokens": 100},
                "timestamp": "2026-01-01T00:00:00+00:00",
            },
        },
    }
    grade_none = {
        "study_name": "gpt-5.2-reasoning-ablation",
        "grader_model": "gpt-4.1",
        "variant": {"id": "gpt-5.2__reasoning-none", "model": "gpt-5.2", "reasoning_effort": "none"},
        "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T00:00:00+00:00",
        "cases": {
            "C1": {
                "case_id": "C1",
                "ground_truth_diagnosis": "A",
                "gold_reasoning_checklist": ["g1"],
                "predicted_diagnosis": "A",
                "diagnosis_correctness_score": 1,
                "diagnosis_correctness_label": "correct",
                "diagnosis_explanation": "good",
                "reasoning_alignment_score": 4,
                "reasoning_alignment_label": "strongly aligned",
                "reasoning_explanation": "good",
                "grader_model": "gpt-4.1",
                "grader_timestamp": "2026-01-01T00:00:00+00:00",
            },
            "C2": {
                "case_id": "C2",
                "ground_truth_diagnosis": "B",
                "gold_reasoning_checklist": ["g2"],
                "predicted_diagnosis": "B",
                "diagnosis_correctness_score": 1,
                "diagnosis_correctness_label": "correct",
                "diagnosis_explanation": "good",
                "reasoning_alignment_score": 3,
                "reasoning_alignment_label": "mostly aligned",
                "reasoning_explanation": "ok",
                "grader_model": "gpt-4.1",
                "grader_timestamp": "2026-01-01T00:00:00+00:00",
            },
        },
    }
    grade_high = {
        "study_name": "gpt-5.2-reasoning-ablation",
        "grader_model": "gpt-4.1",
        "variant": {"id": "gpt-5.2__reasoning-high", "model": "gpt-5.2", "reasoning_effort": "high"},
        "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T00:00:00+00:00",
        "cases": {
            "C1": {
                "case_id": "C1",
                "ground_truth_diagnosis": "A",
                "gold_reasoning_checklist": ["g1"],
                "predicted_diagnosis": "A2",
                "diagnosis_correctness_score": 0,
                "diagnosis_correctness_label": "incorrect",
                "diagnosis_explanation": "wrong",
                "reasoning_alignment_score": 2,
                "reasoning_alignment_label": "mixed",
                "reasoning_explanation": "partial",
                "grader_model": "gpt-4.1",
                "grader_timestamp": "2026-01-01T00:00:00+00:00",
            },
            "C2": {
                "case_id": "C2",
                "ground_truth_diagnosis": "B",
                "gold_reasoning_checklist": ["g2"],
                "predicted_diagnosis": "B",
                "diagnosis_correctness_score": 1,
                "diagnosis_correctness_label": "correct",
                "diagnosis_explanation": "good",
                "reasoning_alignment_score": 3,
                "reasoning_alignment_label": "mostly aligned",
                "reasoning_explanation": "ok",
                "grader_model": "gpt-4.1",
                "grader_timestamp": "2026-01-01T00:00:00+00:00",
            },
        },
    }

    write_json(Path(settings.results_dir) / "gpt-5.2__reasoning-none.json", run_none)
    write_json(Path(settings.results_dir) / "gpt-5.2__reasoning-high.json", run_high)
    write_json(Path(settings.scores_dir) / "gpt-5.2__reasoning-none.json", grade_none)
    write_json(Path(settings.scores_dir) / "gpt-5.2__reasoning-high.json", grade_high)


def test_exact_mcnemar_p_value_is_two_sided():
    assert _mcnemar_exact_p_value(0, 0) == 1.0
    assert _mcnemar_exact_p_value(1, 0) == 1.0
    assert _mcnemar_exact_p_value(4, 0) == 0.125


def test_wilson_interval_bounds():
    low, high = _wilson_interval(successes=8, total=10)
    assert 0 <= low <= high <= 1
    assert low < 0.8 < high


def test_pairwise_and_discordant_exports(tmp_path):
    settings = StudySettings(
        results_dir=str(tmp_path / "results"),
        scores_dir=str(tmp_path / "scores"),
        reports_dir=str(tmp_path / "reports"),
    )
    _seed_files(settings)

    variants = _variant_rows(settings)
    assert [row["reasoning_effort"] for row in variants] == ["none", "high"]

    pairs = _pairwise_rows(settings)
    assert pairs == []

    discordant = export_discordant_cases(settings, a_level="none", b_level="high", limit=10)
    assert len(discordant) == 1
    assert discordant[0]["case_id"] == "C1"
    assert discordant[0]["a_correctness_label"] == "correct"
    assert discordant[0]["b_correctness_label"] == "incorrect"
