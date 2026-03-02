from __future__ import annotations

from pathlib import Path

from .settings import ModelVariant, StudySettings


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def results_dir(settings: StudySettings) -> Path:
    return ensure_dir(Path(settings.results_dir))


def scores_dir(settings: StudySettings) -> Path:
    return ensure_dir(Path(settings.scores_dir))


def reports_dir(settings: StudySettings) -> Path:
    return ensure_dir(Path(settings.reports_dir))


def result_path(settings: StudySettings, variant: ModelVariant) -> Path:
    return results_dir(settings) / f"{variant.variant_id}.json"


def score_path(settings: StudySettings, variant: ModelVariant) -> Path:
    return scores_dir(settings) / f"{variant.variant_id}.json"
