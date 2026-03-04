from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SUPPORTED_REASONING_LEVELS = ("none", "low", "medium", "high")
DEFAULT_REASONING_LEVELS = ("none", "low", "medium", "high")
DEFAULT_DATASET_NAME = "zou-lab/MedCaseReasoning"
DEFAULT_DATASET_SPLIT = "test"
DEFAULT_MODEL_NAME = "gpt-5.2"
DEFAULT_GRADER_MODEL = "gpt-4.1"


@dataclass(frozen=True)
class ModelVariant:
    model: str
    reasoning_effort: str

    @property
    def variant_id(self) -> str:
        return f"{self.model}__reasoning-{self.reasoning_effort}"


@dataclass
class StudySettings:
    dataset_name: str = DEFAULT_DATASET_NAME
    dataset_split: str = DEFAULT_DATASET_SPLIT
    max_cases: int | None = None
    model: str = DEFAULT_MODEL_NAME
    grader_model: str = DEFAULT_GRADER_MODEL
    reasoning_levels: list[str] = field(default_factory=lambda: list(DEFAULT_REASONING_LEVELS))
    max_output_tokens: int = 1200
    grader_max_output_tokens: int = 500
    text_verbosity: str = "low"
    reasoning_summary: str = "concise"
    temperature: float = 0.0
    results_dir: str = "results"
    scores_dir: str = "scores"
    reports_dir: str = "reports"

    @classmethod
    def from_json(cls, path: str | Path | None = None) -> "StudySettings":
        if path is None:
            settings = cls()
        else:
            payload = json.loads(Path(path).read_text())
            settings = cls(**payload)
        settings.validate()
        return settings

    def validate(self) -> None:
        invalid = [level for level in self.reasoning_levels if level not in SUPPORTED_REASONING_LEVELS]
        if invalid:
            raise ValueError(f"Unsupported reasoning levels: {invalid}")
        if self.model != DEFAULT_MODEL_NAME:
            raise ValueError(f"This repository only supports {DEFAULT_MODEL_NAME}, got {self.model!r}")
        if self.grader_model != DEFAULT_GRADER_MODEL:
            raise ValueError(
                f"This repository is configured to use {DEFAULT_GRADER_MODEL} as the grader, got {self.grader_model!r}"
            )
        if self.max_output_tokens <= 0 or self.grader_max_output_tokens <= 0:
            raise ValueError("Token limits must be positive integers.")
        if self.text_verbosity not in {"low", "medium", "high"}:
            raise ValueError("text_verbosity must be one of: low, medium, high")
        if self.reasoning_summary not in {"auto", "concise", "detailed"}:
            raise ValueError("reasoning_summary must be one of: auto, concise, detailed")
        if not 0 <= self.temperature <= 2:
            raise ValueError("temperature must be between 0 and 2 inclusive")

    def variants(self, requested: list[str] | None = None) -> list[ModelVariant]:
        levels = requested or self.reasoning_levels
        for level in levels:
            if level not in SUPPORTED_REASONING_LEVELS:
                raise ValueError(f"Unsupported reasoning level: {level}")
        return [ModelVariant(model=self.model, reasoning_effort=level) for level in levels]

    def as_dict(self) -> dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "dataset_split": self.dataset_split,
            "max_cases": self.max_cases,
            "model": self.model,
            "grader_model": self.grader_model,
            "reasoning_levels": list(self.reasoning_levels),
            "max_output_tokens": self.max_output_tokens,
            "grader_max_output_tokens": self.grader_max_output_tokens,
            "text_verbosity": self.text_verbosity,
            "reasoning_summary": self.reasoning_summary,
            "temperature": self.temperature,
            "results_dir": self.results_dir,
            "scores_dir": self.scores_dir,
            "reports_dir": self.reports_dir,
        }
