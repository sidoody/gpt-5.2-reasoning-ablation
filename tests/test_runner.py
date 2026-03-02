from gpt_5_2_reasoning_ablation.runner import (
    extract_output_json,
    initial_output_tokens,
    max_structured_output_tokens,
    next_retry_output_tokens,
)
from gpt_5_2_reasoning_ablation.settings import ModelVariant, StudySettings


def test_extract_output_json_reads_parsed_block():
    payload = {
        "output": [
            {
                "type": "message",
                "content": [
                    {
                        "type": "output_text",
                        "parsed": {
                            "diagnosis": "Acute myocarditis",
                            "rationale_summary": ["Troponin elevated", "MRI pattern supports diagnosis"],
                        },
                    }
                ],
            }
        ]
    }

    extracted = extract_output_json(payload)
    assert extracted == {
        "diagnosis": "Acute myocarditis",
        "rationale_summary": ["Troponin elevated", "MRI pattern supports diagnosis"],
    }


def test_max_structured_output_tokens_uses_public_retry_ceiling_for_all_variants():
    settings = StudySettings(max_output_tokens=1200)
    xhigh_variant = ModelVariant(model="gpt-5.2", reasoning_effort="xhigh")
    high_variant = ModelVariant(model="gpt-5.2", reasoning_effort="high")

    assert max_structured_output_tokens(settings, xhigh_variant) == 9600
    assert max_structured_output_tokens(settings, high_variant) == 9600


def test_initial_output_tokens_starts_from_configured_base_budget():
    settings = StudySettings(max_output_tokens=1200)
    xhigh_variant = ModelVariant(model="gpt-5.2", reasoning_effort="xhigh")
    medium_variant = ModelVariant(model="gpt-5.2", reasoning_effort="medium")

    assert initial_output_tokens(settings, xhigh_variant) == 1200
    assert initial_output_tokens(settings, medium_variant) == 1200


def test_next_retry_output_tokens_clamps_to_retry_ceiling():
    assert next_retry_output_tokens(1200, 9600) == 2400
    assert next_retry_output_tokens(4800, 9600) == 9600
    assert next_retry_output_tokens(9600, 9600) == 9600

