from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


class ModelVisibleAnswer(BaseModel):
    diagnosis: str = Field(..., description="Single best final diagnosis.")
    rationale_summary: list[str] = Field(
        default_factory=list,
        description="Short, grader-visible rationale bullets grounded in the case details.",
    )

    @field_validator("diagnosis")
    @classmethod
    def diagnosis_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("diagnosis must not be blank")
        return value

    @field_validator("rationale_summary")
    @classmethod
    def trim_rationale_items(cls, value: list[str]) -> list[str]:
        cleaned = [item.strip() for item in value if item and item.strip()]
        return cleaned[:8]


class CaseResult(BaseModel):
    case_id: str
    diagnosis: str
    rationale_summary: list[str] = Field(default_factory=list)
    raw_output_text: str
    parsed_output: dict[str, Any]
    api_reasoning_summary: str | None = None
    latency_seconds: float
    usage: dict[str, Any] = Field(default_factory=dict)
    timestamp: str


class RunFile(BaseModel):
    study_name: str
    dataset: dict[str, Any]
    variant: dict[str, Any]
    run_settings: dict[str, Any]
    created_at: str
    updated_at: str
    cases: dict[str, CaseResult] = Field(default_factory=dict)


class GradeOutcome(BaseModel):
    case_id: str
    ground_truth_diagnosis: str
    gold_reasoning_checklist: list[str]
    predicted_diagnosis: str
    diagnosis_correctness_score: int = Field(ge=0, le=1)
    diagnosis_correctness_label: str
    diagnosis_explanation: str
    reasoning_alignment_score: int = Field(ge=0, le=4)
    reasoning_alignment_label: str
    reasoning_explanation: str
    grader_model: str
    grader_timestamp: str


class GradeFile(BaseModel):
    study_name: str
    grader_model: str
    variant: dict[str, Any]
    created_at: str
    updated_at: str
    cases: dict[str, GradeOutcome] = Field(default_factory=dict)
