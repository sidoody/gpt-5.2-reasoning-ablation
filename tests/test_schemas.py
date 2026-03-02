import pytest
from pydantic import ValidationError

from gpt_5_2_reasoning_ablation.schemas import GradeOutcome


def _base_grade_outcome_payload() -> dict:
    return {
        "case_id": "PMC123",
        "ground_truth_diagnosis": "myocarditis",
        "gold_reasoning_checklist": ["troponin elevation", "cardiac MRI pattern"],
        "predicted_diagnosis": "acute myocarditis",
        "diagnosis_correctness_score": 1,
        "diagnosis_correctness_label": "correct",
        "diagnosis_explanation": "The predicted diagnosis is clinically consistent with the gold diagnosis.",
        "reasoning_alignment_score": 3,
        "reasoning_alignment_label": "mostly aligned",
        "reasoning_explanation": "Main signals are present.",
        "grader_model": "gpt-4.1",
        "grader_timestamp": "2026-01-01T00:00:00Z",
    }


def test_grade_outcome_rejects_diagnosis_score_above_one():
    payload = _base_grade_outcome_payload()
    payload["diagnosis_correctness_score"] = 2
    with pytest.raises(ValidationError):
        GradeOutcome.model_validate(payload)


def test_grade_outcome_rejects_diagnosis_score_below_zero():
    payload = _base_grade_outcome_payload()
    payload["diagnosis_correctness_score"] = -1
    with pytest.raises(ValidationError):
        GradeOutcome.model_validate(payload)


def test_grade_outcome_rejects_reasoning_score_above_four():
    payload = _base_grade_outcome_payload()
    payload["reasoning_alignment_score"] = 5
    with pytest.raises(ValidationError):
        GradeOutcome.model_validate(payload)
