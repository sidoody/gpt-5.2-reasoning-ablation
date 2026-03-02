from __future__ import annotations

from .case import ClinicalCase

MODEL_INSTRUCTIONS = """You are participating in a clinical reasoning benchmark.
Return exactly one best diagnosis and a short, grader-visible rationale summary.
Do not include hidden chain-of-thought, extra commentary, markdown, or prose outside the JSON schema.
Ground the rationale only in case facts that are actually present in the prompt.
"""

GRADER_INSTRUCTIONS = """You are grading a clinical reasoning benchmark built from case reports (rare-disease-skewed, not general-population triage).
Evaluate two separate dimensions:
1. Diagnosis correctness: whether the predicted diagnosis is clinically correct relative to the gold diagnosis.
2. Reasoning alignment: whether the visible rationale relies on the decisive clinical reasons compared with the gold reasoning rubric bullets.

Scoring rubric:
- diagnosis_correctness_score:
  1 = correct
  0 = incorrect
- reasoning_alignment_score:
  4 = strongly aligned
  3 = mostly aligned
  2 = mixed
  1 = poorly aligned
  0 = poorly aligned

Reasoning guidance:
- Emphasize decisive diagnostic clues and key exclusions when clinically important.
- Prioritize whether the visible rationale actually justifies the predicted diagnosis.
- Judge semantic equivalence of rationale points, not string overlap with the rubric.
- Do not over-penalize harmless extra detail if the decisive rationale is correct.
- Missing minor paper-specific differentials should not heavily reduce an otherwise strong score.
- Do not inflate score when the answer misses a decisive discriminator, reaches the diagnosis for the wrong main reason, or ignores a crucial rule-out finding.

Judge only the visible answer provided. Do not assume access to hidden reasoning.
Respond only with the JSON schema.
"""


def build_case_input(case: ClinicalCase) -> str:
    return f"Case ID: {case.pmcid}\n\nClinical case:\n{case.case_prompt.strip()}"


def build_grader_input(
    case: ClinicalCase,
    predicted_diagnosis: str,
    raw_answer_text: str,
    visible_rationale: list[str],
) -> str:
    checklist = case.reasoning_rubric()
    checklist_text = "\n".join(f"- {item}" for item in checklist) if checklist else "- No rubric bullets available"
    rationale_text = "\n".join(f"- {item}" for item in visible_rationale) if visible_rationale else "- No visible rationale provided"

    return (
        f"Case ID: {case.pmcid}\n\n"
        f"Case prompt:\n{case.case_prompt.strip()}\n\n"
        f"Gold diagnosis:\n{case.final_diagnosis.strip()}\n\n"
        f"Gold reasoning rubric bullets:\n{checklist_text}\n\n"
        f"Predicted diagnosis:\n{predicted_diagnosis.strip()}\n\n"
        f"Model visible answer:\n{raw_answer_text.strip()}\n\n"
        f"Visible rationale summary:\n{rationale_text}"
    )
