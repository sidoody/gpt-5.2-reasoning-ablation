from __future__ import annotations

import re
from pydantic import BaseModel

from .text_utils import normalize_text

_INCOMPLETE_ENDINGS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "because",
    "by",
    "due",
    "for",
    "from",
    "in",
    "is",
    "of",
    "or",
    "the",
    "to",
    "was",
    "were",
    "with",
}


def build_gold_reasoning_rubric(
    diagnostic_reasoning: str,
    *,
    min_items: int = 3,
    max_items: int = 6,
) -> list[str]:
    text = normalize_text(diagnostic_reasoning)
    if not text or max_items <= 0:
        return []

    # Break inline numbered lists into separate candidate fragments.
    text = re.sub(r'(?:(?<=\s)|(?<=^)|(?<=[\'"]))(?=\d+[\).:-]\s+)', "\n", text)
    lines = [line.strip() for line in re.split(r"\n+", text) if line.strip()]
    fragments: list[str] = []
    for line in lines:
        pieces = re.split(r"(?<=[.!?;])\s+", line)
        fragments.extend(piece for piece in pieces if piece.strip())

    def normalize_fragment(fragment: str, min_len: int = 8) -> str:
        item = normalize_text(fragment).strip(" \t\r\n-*•\"'`")
        item = re.sub(r"^\(?\d+[\).:-]\s*", "", item)
        item = re.sub(r"\[(?:\d+\s*[,-]?\s*)+\]", "", item)
        item = re.sub(r'\s*[\-\u2013\u2014]\s*["\'].*$', "", item)
        if '"' in item:
            item = item.split('"', 1)[0]
        if "'" in item and len(item.split("'", 1)[0]) >= min_len:
            item = item.split("'", 1)[0]
        item = re.sub(r"\s+", " ", item).strip(" ,;:-")
        if len(item) < min_len:
            return ""
        if len(item) > 220:
            item = item[:220].rsplit(" ", 1)[0]
        last_word = item.rsplit(" ", 1)[-1].lower()
        if last_word in _INCOMPLETE_ENDINGS:
            return ""
        if item.endswith(":") or item.endswith("-"):
            return ""
        return item.rstrip(" ,;:-")

    rubric: list[str] = []
    seen: set[str] = set()

    for fragment in fragments:
        item = normalize_fragment(fragment)
        if not item:
            continue
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        rubric.append(item)
        if len(rubric) >= max_items:
            return rubric

    # If the source text is sparse, salvage compact comma-delimited clues.
    if len(rubric) < min_items:
        for fragment in re.split(r",\s+", text):
            item = normalize_fragment(fragment, min_len=8)
            if not item:
                continue
            key = item.lower()
            if key in seen:
                continue
            seen.add(key)
            rubric.append(item)
            if len(rubric) >= max_items:
                break

    return rubric[:max_items]


class ClinicalCase(BaseModel):
    pmcid: str
    journal: str = ""
    article_link: str = ""
    case_prompt: str
    diagnostic_reasoning: str
    final_diagnosis: str

    def reasoning_rubric(self, max_items: int = 6) -> list[str]:
        return build_gold_reasoning_rubric(
            self.diagnostic_reasoning,
            min_items=3,
            max_items=max_items,
        )

    def reasoning_checklist(self, max_items: int = 6) -> list[str]:
        return self.reasoning_rubric(max_items=max_items)

    def grading_reference(self) -> dict:
        return {
            "gold_diagnosis": self.final_diagnosis,
            "gold_reasoning_checklist": self.reasoning_rubric(),
        }


class CaseLibrary(BaseModel):
    cases: list[ClinicalCase]

    @classmethod
    def from_huggingface(
        cls,
        dataset_name: str = "zou-lab/MedCaseReasoning",
        split: str = "test",
    ) -> "CaseLibrary":
        from datasets import load_dataset

        dataset = load_dataset(dataset_name)
        cases = [
            ClinicalCase(
                pmcid=row["pmcid"],
                journal=row.get("journal", ""),
                article_link=row.get("article_link", ""),
                case_prompt=row["case_prompt"],
                diagnostic_reasoning=row["diagnostic_reasoning"],
                final_diagnosis=row["final_diagnosis"],
            )
            for row in dataset[split]
        ]
        return cls(cases=cases)

    def limited(self, limit: int | None) -> "CaseLibrary":
        if limit is None:
            return self
        return CaseLibrary(cases=self.cases[:limit])

    def by_pmcid(self) -> dict[str, ClinicalCase]:
        return {case.pmcid: case for case in self.cases}

    def __len__(self) -> int:
        return len(self.cases)

    def __iter__(self):
        return iter(self.cases)
