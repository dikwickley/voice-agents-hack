"""Lightweight document risk tagging + extractive summary (no extra LLM)."""

from __future__ import annotations

import re
from typing import Any


_RISK_PATTERNS = [
    (r"\b(?:penalty|fines?|liability|indemnif)\b", "legal_financial"),
    (r"\b(?:SSN|social security|credit card|\b\d{3}-\d{2}-\d{4}\b)\b", "pii"),
    (r"\b(?:password|secret key|api[_-]?key|bearer\s+token)\b", "secrets"),
    (r"\b(?:HIPAA|GDPR|PCI|SOC\s*2)\b", "compliance"),
    (r"\b(?:urgent|immediately|ASAP|deadline)\b", "urgency"),
    (r"\b(?:guarantee|warranty|unlimited)\b", "marketing_claim"),
]


def run(args: dict[str, Any]) -> dict[str, Any]:
    text = (args.get("text") or "").strip()
    if not text:
        return {"error": "empty text"}

    risks: list[dict[str, str]] = []
    lower = text.lower()
    for pat, tag in _RISK_PATTERNS:
        if re.search(pat, text, re.I):
            risks.append({"tag": tag, "snippet": _snippet(text, pat)})

    summary = _summarize(text)
    return {
        "summary": summary,
        "risks": risks[:12],
        "char_count": len(text),
    }


def _snippet(text: str, pat: str) -> str:
    m = re.search(pat, text, re.I)
    if not m:
        return ""
    start = max(0, m.start() - 40)
    end = min(len(text), m.end() + 40)
    return text[start:end].replace("\n", " ")


def _summarize(text: str, max_sentences: int = 3) -> str:
    parts = re.split(r"(?<=[.!?])\s+", text)
    sentences = [p.strip() for p in parts if len(p.strip()) > 20]
    if not sentences:
        return text[:400] + ("…" if len(text) > 400 else "")
    return " ".join(sentences[:max_sentences])
