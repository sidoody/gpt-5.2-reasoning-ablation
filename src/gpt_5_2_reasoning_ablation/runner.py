from __future__ import annotations

import json
import time
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import ValidationError

from .case import CaseLibrary
from .io_utils import read_json, utc_now_iso, write_json
from .paths import result_path
from .prompts import MODEL_INSTRUCTIONS, build_case_input
from .schemas import CaseResult, ModelVisibleAnswer, RunFile
from .settings import ModelVariant, StudySettings
from .text_utils import normalize_text, normalize_text_list

load_dotenv()

MODEL_OUTPUT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "diagnosis": {"type": "string"},
        "rationale_summary": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "maxItems": 8,
        },
    },
    "required": ["diagnosis", "rationale_summary"],
}

MAX_STRUCTURED_OUTPUT_TOKENS = 9600


def _response_to_dict(response: Any) -> dict[str, Any]:
    if hasattr(response, "model_dump"):
        return response.model_dump()
    if isinstance(response, dict):
        return response
    raise TypeError(f"Unsupported response type: {type(response)!r}")


def extract_output_text(response_payload: dict[str, Any]) -> str:
    output_text = response_payload.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text

    parts: list[str] = []
    for item in response_payload.get("output", []):
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") not in {"output_text", "text"}:
                continue
            text = block.get("text")
            if isinstance(text, str) and text.strip():
                parts.append(text)

    return "\n".join(parts).strip()


def extract_output_json(response_payload: dict[str, Any]) -> dict[str, Any] | None:
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
    return None


def max_structured_output_tokens(settings: StudySettings, variant: ModelVariant) -> int:
    _ = variant
    return max(settings.max_output_tokens, MAX_STRUCTURED_OUTPUT_TOKENS)


def initial_output_tokens(settings: StudySettings, variant: ModelVariant) -> int:
    _ = variant
    return settings.max_output_tokens


def next_retry_output_tokens(current_limit: int, max_token_limit: int) -> int:
    return min(current_limit * 2, max_token_limit)


def extract_reasoning_summary(response_payload: dict[str, Any]) -> str | None:
    parts: list[str] = []
    for item in response_payload.get("output", []):
        if item.get("type") != "reasoning":
            continue
        summary = item.get("summary")
        if isinstance(summary, list):
            for entry in summary:
                if isinstance(entry, dict):
                    text = entry.get("text") or entry.get("summary")
                    if text:
                        parts.append(str(text).strip())
                elif isinstance(entry, str):
                    parts.append(entry.strip())
        elif isinstance(summary, str):
            parts.append(summary.strip())
    joined = " ".join(part for part in parts if part)
    return joined or None


def extract_usage(response_payload: dict[str, Any]) -> dict[str, Any]:
    usage = response_payload.get("usage") or {}
    details = usage.get("output_tokens_details") or {}
    return {
        "input_tokens": usage.get("input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0),
        "reasoning_tokens": details.get("reasoning_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
    }


def create_empty_run(settings: StudySettings, variant: ModelVariant) -> RunFile:
    now = utc_now_iso()
    return RunFile(
        study_name="gpt-5.2-reasoning-ablation",
        dataset={
            "name": settings.dataset_name,
            "split": settings.dataset_split,
        },
        variant={
            "id": variant.variant_id,
            "model": variant.model,
            "reasoning_effort": variant.reasoning_effort,
        },
        run_settings={
            "max_output_tokens": settings.max_output_tokens,
            "text_verbosity": settings.text_verbosity,
            "reasoning_summary": settings.reasoning_summary,
            "temperature": settings.temperature,
        },
        created_at=now,
        updated_at=now,
        cases={},
    )


def load_or_initialize_run(settings: StudySettings, variant: ModelVariant, overwrite: bool = False) -> RunFile:
    path = result_path(settings, variant)
    if not overwrite:
        payload = read_json(path)
        if payload:
            run = RunFile.model_validate(payload)
            for case_result in run.cases.values():
                case_result.diagnosis = normalize_text(case_result.diagnosis)
                case_result.rationale_summary = normalize_text_list(case_result.rationale_summary)
                case_result.raw_output_text = normalize_text(case_result.raw_output_text)
                case_result.api_reasoning_summary = (
                    normalize_text(case_result.api_reasoning_summary) if case_result.api_reasoning_summary else None
                )
                if isinstance(case_result.parsed_output, dict):
                    diagnosis = case_result.parsed_output.get("diagnosis")
                    if isinstance(diagnosis, str):
                        case_result.parsed_output["diagnosis"] = normalize_text(diagnosis)
                    rationale_summary = case_result.parsed_output.get("rationale_summary")
                    if isinstance(rationale_summary, list):
                        case_result.parsed_output["rationale_summary"] = normalize_text_list(
                            [str(item) for item in rationale_summary]
                        )
            write_json(path, run.model_dump(mode="json"))
            return run
    run = create_empty_run(settings, variant)
    write_json(path, run.model_dump(mode="json"))
    return run


def call_variant(
    client: OpenAI,
    settings: StudySettings,
    variant: ModelVariant,
    case_prompt: str,
    max_output_tokens: int | None = None,
) -> dict[str, Any]:
    output_token_limit = max_output_tokens or settings.max_output_tokens
    response = client.responses.create(
        model=variant.model,
        instructions=MODEL_INSTRUCTIONS,
        input=case_prompt,
        temperature=settings.temperature,
        max_output_tokens=output_token_limit,
        truncation="disabled",
        store=False,
        reasoning={
            "effort": variant.reasoning_effort,
            "summary": settings.reasoning_summary,
        },
        text={
            "verbosity": settings.text_verbosity,
            "format": {
                "type": "json_schema",
                "name": "clinical_case_answer",
                "strict": True,
                "schema": MODEL_OUTPUT_SCHEMA,
            },
        },
    )
    payload = _response_to_dict(response)
    return payload


def run_variants(
    settings: StudySettings,
    requested_variants: list[str] | None = None,
    limit: int | None = None,
    overwrite: bool = False,
) -> None:
    client = OpenAI()
    library = CaseLibrary.from_huggingface(
        settings.dataset_name,
        settings.dataset_split,
    ).limited(limit or settings.max_cases)

    for variant in settings.variants(requested_variants):
        run = load_or_initialize_run(settings, variant, overwrite=overwrite)
        path = result_path(settings, variant)
        print(f"\n=== Running {variant.variant_id} on {len(library)} cases ===")

        for index, case in enumerate(library, start=1):
            if not overwrite and case.pmcid in run.cases:
                continue

            start = time.perf_counter()
            case_prompt = build_case_input(case)
            token_limit = initial_output_tokens(settings, variant)
            max_token_limit = max_structured_output_tokens(settings, variant)
            payload: dict[str, Any] | None = None
            raw_output_text = ""
            parsed_output: dict[str, Any] | None = None
            visible_answer: ModelVisibleAnswer | None = None
            attempt = 1
            print(
                f"[{index}/{len(library)}] Starting case {case.pmcid} "
                f"(attempt {attempt}, max_output_tokens={token_limit})",
                flush=True,
            )

            while True:
                payload = call_variant(
                    client,
                    settings,
                    variant,
                    case_prompt,
                    max_output_tokens=token_limit,
                )
                raw_output_text = extract_output_text(payload)
                output_json = extract_output_json(payload)
                try:
                    if output_json is not None:
                        visible_answer = ModelVisibleAnswer.model_validate(output_json)
                    else:
                        visible_answer = ModelVisibleAnswer.model_validate_json(raw_output_text)
                    visible_answer = ModelVisibleAnswer(
                        diagnosis=normalize_text(visible_answer.diagnosis),
                        rationale_summary=normalize_text_list(visible_answer.rationale_summary),
                    )
                    parsed_output = visible_answer.model_dump(mode="json")
                    normalized_raw_output_text = json.dumps(parsed_output, ensure_ascii=False)
                    break
                except ValidationError as exc:
                    incomplete_details = payload.get("incomplete_details") or {}
                    if (
                        payload.get("status") == "incomplete"
                        and incomplete_details.get("reason") == "max_output_tokens"
                    ):
                        if token_limit >= max_token_limit:
                            raise RuntimeError(
                                f"Structured output exhausted max_output_tokens for case {case.pmcid}.\n"
                                f"Token limit used: {token_limit}\n"
                                f"Configured retry ceiling: {max_token_limit}\n"
                                f"Response status: {payload.get('status')}\n"
                                f"Incomplete details: {payload.get('incomplete_details')}\n"
                                f"Response error: {payload.get('error')}\n"
                                f"Raw output: {raw_output_text!r}"
                            ) from exc
                        previous_limit = token_limit
                        token_limit = next_retry_output_tokens(token_limit, max_token_limit)
                        attempt += 1
                        print(
                            f"[{index}/{len(library)}] Retrying case {case.pmcid} "
                            f"(attempt {attempt}, max_output_tokens={token_limit}, "
                            f"previous_limit={previous_limit})",
                            flush=True,
                        )
                        continue
                    raise RuntimeError(
                        f"Structured output validation failed for case {case.pmcid}: {exc}\n"
                        f"Response status: {payload.get('status')}\n"
                        f"Incomplete details: {payload.get('incomplete_details')}\n"
                        f"Response error: {payload.get('error')}\n"
                        f"Raw output: {raw_output_text!r}"
                    ) from exc

            if payload is None or visible_answer is None or parsed_output is None:
                raise RuntimeError(f"No usable response for case {case.pmcid}")
            latency_seconds = round(time.perf_counter() - start, 3)

            run.cases[case.pmcid] = CaseResult(
                case_id=case.pmcid,
                diagnosis=visible_answer.diagnosis,
                rationale_summary=visible_answer.rationale_summary,
                raw_output_text=normalized_raw_output_text,
                parsed_output=parsed_output,
                api_reasoning_summary=normalize_text(extract_reasoning_summary(payload) or "")
                or None,
                latency_seconds=latency_seconds,
                usage=extract_usage(payload),
                timestamp=utc_now_iso(),
            )
            run.updated_at = utc_now_iso()
            write_json(path, run.model_dump(mode="json"))
            print(f"[{index}/{len(library)}] {case.pmcid} -> {visible_answer.diagnosis[:80]}")
