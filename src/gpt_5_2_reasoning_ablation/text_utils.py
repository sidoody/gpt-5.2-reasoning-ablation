from __future__ import annotations

import re

_ESCAPED_UNICODE_RE = re.compile(r"\\u([0-9a-fA-F]{4})")
_ESCAPED_BYTE_RE = re.compile(r"\\x([0-9a-fA-F]{2})")


def normalize_text(text: str) -> str:
    if not text:
        return ""

    text = text.replace('\\"', '"').replace("\\'", "'")
    text = text.replace("\\n", " ").replace("\\t", " ")

    text = _ESCAPED_UNICODE_RE.sub(lambda match: chr(int(match.group(1), 16)), text)
    text = _ESCAPED_BYTE_RE.sub(lambda match: chr(int(match.group(1), 16)), text)

    text = (
        text.replace("\u2014", "-")
        .replace("\u2013", "-")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2018", "'")
        .replace("\u2019", "'")
    )
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_text_list(values: list[str]) -> list[str]:
    cleaned: list[str] = []
    for value in values:
        item = normalize_text(value)
        if item:
            cleaned.append(item)
    return cleaned
