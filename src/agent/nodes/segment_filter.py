"""
Node: segment_filter

Detects a reviewer segment from the query text. Runs after query_classifier —
if the classifier already extracted a segment, this node passes it through.
If not, keyword detection is used as a fallback.

segment=None downstream means "no filter applied".

Reads:  query, segment
Writes: segment
"""

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
