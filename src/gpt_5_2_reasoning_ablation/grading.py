from __future__ import annotations

from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError

from .case import CaseLibrary
from .io_utils import read_json, utc_now_iso, write_json
from .paths import result_path, score_path
from .prompts import GRADER_INSTRUCTIONS, build_grader_input
from .schemas import GradeFile, GradeOutcome, RunFile
from .settings import ModelVariant, StudySettings
from .text_utils import normalize_text, normalize_text_list

load_dotenv()


class GradeResponse(BaseModel):
    diagnosis_correctness_score: int = Field(ge=0, le=1)
    diagnosis_correctness_label: str
    diagnosis_explanation: str
    reasoning_alignment_score: int = Field(ge=0, le=4)
    reasoning_alignment_label: str
    reasoning_explanation: str


GRADE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "diagnosis_correctness_score": {"type": "integer", "minimum": 0, "maximum": 1},
        "diagnosis_correctness_label": {"type": "string"},
        "diagnosis_explanation": {"type": "string"},
        "reasoning_alignment_score": {"type": "integer", "minimum": 0, "maximum": 4},
        "reasoning_alignment_label": {"type": "string"},
        "reasoning_explanation": {"type": "string"},
    },
    "required": [
        "diagnosis_correctness_score",
        "diagnosis_correctness_label",
        "diagnosis_explanation",
        "reasoning_alignment_score",
        "reasoning_alignment_label",
        "reasoning_explanation",
    ],
}

DIAGNOSIS_LABELS = {
    1: "correct",
    0: "incorrect",
}

REASONING_LABELS = {
    4: "strongly aligned",
    3: "mostly aligned",
    2: "mixed",
    1: "poorly aligned",
    0: "poorly aligned",
}


def _response_to_dict(response: Any) -> dict[str, Any]:
    if hasattr(response, "model_dump"):
        return response.model_dump()
    if isinstance(response, dict):
        return response
    raise TypeError(f"Unsupported response type: {type(response)!r}")


def _extract_grade_payload(response_payload: dict[str, Any]) -> dict[str, Any] | str | None:
    output_text = response_payload.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text

    for item in response_payload.get("output", []):
        if not isinstance(item, dict) or item.get("type") != "message":
            continue

        content = item.get("content")
        if not isinstance(content, list):
            continue

        for block in content:
            if not isinstance(block, dict):
                continue

            parsed = block.get("parsed")
            if isinstance(parsed, dict):
                return parsed

            json_value = block.get("json")
            if isinstance(json_value, dict):
                return json_value

            text = block.get("text")
            if isinstance(text, str) and text.strip():
                return text

    return None


def create_empty_grade_file(settings: StudySettings, variant: ModelVariant) -> GradeFile:
    now = utc_now_iso()
    return GradeFile(
        study_name="gpt-5.2-reasoning-ablation",
        grader_model=settings.grader_model,
        variant={
            "id": variant.variant_id,
            "model": variant.model,
            "reasoning_effort": variant.reasoning_effort,
        },
        created_at=now,
        updated_at=now,
        cases={},
    )


def load_or_initialize_grade_file(settings: StudySettings, variant: ModelVariant, overwrite: bool = False) -> GradeFile:
    path = score_path(settings, variant)
    if not overwrite:
        payload = read_json(path)
        if payload:
            grade_file = GradeFile.model_validate(payload)
            for outcome in grade_file.cases.values():
                outcome.ground_truth_diagnosis = normalize_text(outcome.ground_truth_diagnosis)
                outcome.gold_reasoning_checklist = normalize_text_list(outcome.gold_reasoning_checklist)
                outcome.predicted_diagnosis = normalize_text(outcome.predicted_diagnosis)
                outcome.diagnosis_correctness_label = DIAGNOSIS_LABELS[outcome.diagnosis_correctness_score]
                outcome.diagnosis_explanation = normalize_text(outcome.diagnosis_explanation)
                outcome.reasoning_alignment_label = REASONING_LABELS[outcome.reasoning_alignment_score]
                outcome.reasoning_explanation = normalize_text(outcome.reasoning_explanation)
            write_json(path, grade_file.model_dump(mode="json"))
            return grade_file
    grade_file = create_empty_grade_file(settings, variant)
    write_json(path, grade_file.model_dump(mode="json"))
    return grade_file


def grade_one_case(client: OpenAI, settings: StudySettings, grader_input: str) -> GradeResponse:
    token_limit = settings.grader_max_output_tokens

    for _ in range(3):
        response = client.responses.create(
            model=settings.grader_model,
            instructions=GRADER_INSTRUCTIONS,
            input=grader_input,
            temperature=settings.temperature,
            max_output_tokens=token_limit,
            truncation="disabled",
            store=False,
            text={
                "verbosity": "medium",
                "format": {
                    "type": "json_schema",
                    "name": "clinical_case_grade",
                    "strict": True,
                    "schema": GRADE_SCHEMA,
                },
            },
        )
        payload = _response_to_dict(response)
        grade_payload = _extract_grade_payload(payload)

        try:
            if isinstance(grade_payload, dict):
                return GradeResponse.model_validate(grade_payload)
            if isinstance(grade_payload, str):
                return GradeResponse.model_validate_json(grade_payload)
        except ValidationError:
            pass

        incomplete_details = payload.get("incomplete_details") or {}
        if (
            payload.get("status") == "incomplete"
            and incomplete_details.get("reason") == "max_output_tokens"
            and token_limit < 4000
        ):
            token_limit *= 2
            continue

        raise RuntimeError(
            "Failed to extract grader JSON response. "
            f"status={payload.get('status')!r} "
            f"incomplete_details={payload.get('incomplete_details')!r} "
            f"error={payload.get('error')!r} "
            f"raw_payload={grade_payload!r}"
        )

    raise RuntimeError("Failed to extract grader JSON response after retries.")


def grade_variants(
    settings: StudySettings,
    requested_variants: list[str] | None = None,
    overwrite: bool = False,
) -> None:
    client = OpenAI()
    cases = CaseLibrary.from_huggingface(settings.dataset_name, settings.dataset_split).by_pmcid()

    for variant in settings.variants(requested_variants):
        result_payload = read_json(result_path(settings, variant))
        if not result_payload:
            print(f"No results found for {variant.variant_id}; skipping.")
            continue

        run_file = RunFile.model_validate(result_payload)
        grade_file = load_or_initialize_grade_file(settings, variant, overwrite=overwrite)
        destination = score_path(settings, variant)

        print(f"\n=== Grading {variant.variant_id} with {settings.grader_model} ===")

        total_cases = len(run_file.cases)
        already_done = 0 if overwrite else sum(1 for case_id in run_file.cases if case_id in grade_file.cases)
        if already_done:
            next_index = already_done + 1
            if already_done >= total_cases:
                print(f"All {total_cases} cases already graded; nothing to do.")
            else:
                print(
                    f"Skipping {already_done} already graded cases; "
                    f"resuming at [{next_index}/{total_cases}]."
                )
        for index, (case_id, result) in enumerate(run_file.cases.items(), start=1):
            case = cases[case_id]
            if not overwrite and case_id in grade_file.cases:
                existing = grade_file.cases[case_id]
                existing.ground_truth_diagnosis = normalize_text(case.final_diagnosis)
                existing.gold_reasoning_checklist = normalize_text_list(case.reasoning_rubric())
                existing.predicted_diagnosis = normalize_text(result.diagnosis)
                existing.diagnosis_correctness_label = DIAGNOSIS_LABELS[existing.diagnosis_correctness_score]
                existing.diagnosis_explanation = normalize_text(existing.diagnosis_explanation)
                existing.reasoning_alignment_label = REASONING_LABELS[existing.reasoning_alignment_score]
                existing.reasoning_explanation = normalize_text(existing.reasoning_explanation)
                grade_file.updated_at = utc_now_iso()
                write_json(destination, grade_file.model_dump(mode="json"))
                continue

            visible_rationale = list(result.rationale_summary)
            if result.api_reasoning_summary:
                visible_rationale = [*visible_rationale, f"API reasoning summary: {result.api_reasoning_summary}"]

            grade = grade_one_case(
                client,
                settings,
                build_grader_input(
                    case=case,
                    predicted_diagnosis=result.diagnosis,
                    raw_answer_text=result.raw_output_text,
                    visible_rationale=visible_rationale,
                ),
            )

            grade_file.cases[case_id] = GradeOutcome(
                case_id=case_id,
                ground_truth_diagnosis=normalize_text(case.final_diagnosis),
                gold_reasoning_checklist=normalize_text_list(case.reasoning_rubric()),
                predicted_diagnosis=normalize_text(result.diagnosis),
                diagnosis_correctness_score=grade.diagnosis_correctness_score,
                diagnosis_correctness_label=DIAGNOSIS_LABELS[grade.diagnosis_correctness_score],
                diagnosis_explanation=normalize_text(grade.diagnosis_explanation),
                reasoning_alignment_score=grade.reasoning_alignment_score,
                reasoning_alignment_label=REASONING_LABELS[grade.reasoning_alignment_score],
                reasoning_explanation=normalize_text(grade.reasoning_explanation),
                grader_model=settings.grader_model,
                grader_timestamp=utc_now_iso(),
            )
            grade_file.updated_at = utc_now_iso()
            write_json(destination, grade_file.model_dump(mode="json"))
            print(
                f"[{index}/{total_cases}] {case_id}: diag={grade.diagnosis_correctness_score} "
                f"reasoning={grade.reasoning_alignment_score}"
            )
