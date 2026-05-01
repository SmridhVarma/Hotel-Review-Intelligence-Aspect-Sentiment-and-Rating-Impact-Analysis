# segment_filter.py — Stage 5 | Module C (Agent): Detects reviewer segment from query; passes through classifier result or falls back to keyword detection.
#
# Input:  AgentState — query, segment
# Output: AgentState — segment

from __future__ import annotations

import re

from agent.state import AgentState

SEGMENT_KEYWORDS: list[tuple[str, str]] = [
    (r"business travell?ers?",  "Business"),
    (r"business guests?",       "Business"),
    (r"business trip",          "Business"),
    (r"on business",            "Business"),
    (r"couples?",               "Couple"),
    (r"romantic",               "Couple"),
    (r"honeymoon",              "Couple"),
    (r"famil(?:y|ies)",         "Family"),
    (r"with kids?",             "Family"),
    (r"with children",          "Family"),
    (r"solo travell?ers?",      "Solo"),
    (r"travelling alone",       "Solo"),
    (r"by (them|him|her)self",  "Solo"),
    (r"groups?",                "Group"),
    (r"group (booking|travel)", "Group"),
]

_PATTERNS = [(re.compile(pat, re.IGNORECASE), seg) for pat, seg in SEGMENT_KEYWORDS]


def _detect_segment(query: str) -> str | None:
    for pattern, segment in _PATTERNS:
        if pattern.search(query):
            return segment
    return None


def segment_filter(state: AgentState) -> dict:
    segment = state.get("segment") or _detect_segment(state.get("query", ""))
    return {"segment": segment}
