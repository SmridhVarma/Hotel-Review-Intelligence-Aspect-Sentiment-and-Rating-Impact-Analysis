"""
Node: query_classifier

First node in the DAG. Classifies the incoming query into structured fields
using GPT-4o with Pydantic validation. Also resolves the hotel name via fuzzy
matching against the known hotel list.

Reads:  query, hotel_name, conversation_history
Writes: query_type, query_direction, aspects, segment,
        hotel_name, hotel_confidence, hotel_unresolved
"""

from __future__ import annotations

import json
import os
import sys
from typing import Optional

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, field_validator

_SRC = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from agent.prompts import CLASSIFIER_PROMPT
from agent.state import AgentState
from paths import OUTPUT_DIR

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

HOTEL_RESOLUTION_THRESHOLD = 70
VALID_ASPECTS    = {"Cleanliness", "Staff", "Location", "Noise", "Food", "Room"}
VALID_QUERY_TYPES = {"evidence", "prioritization", "mismatch"}
VALID_DIRECTIONS  = {"positive", "negative", "neutral"}
VALID_SEGMENTS    = {"Business", "Couple", "Family", "Solo", "Group"}


# ── Pydantic output schema ────────────────────────────────────────────────────

class ClassifierOutput(BaseModel):
    query_type:      str          = "evidence"
    query_direction: str          = "neutral"
    aspects:         list[str]    = []
    hotel_name:      str          = "__global__"
    segment:         Optional[str] = None

    @field_validator("query_type")
    @classmethod
    def validate_query_type(cls, v: str) -> str:
        return v if v in VALID_QUERY_TYPES else "evidence"

    @field_validator("query_direction")
    @classmethod
    def validate_direction(cls, v: str) -> str:
        return v if v in VALID_DIRECTIONS else "neutral"

    @field_validator("aspects")
    @classmethod
    def validate_aspects(cls, v: list) -> list:
        return [a for a in v if a in VALID_ASPECTS]

    @field_validator("segment")
    @classmethod
    def validate_segment(cls, v: Optional[str]) -> Optional[str]:
        return v if v in VALID_SEGMENTS else None


# ── Hotel name list ───────────────────────────────────────────────────────────

_hotel_names: list[str] | None = None


def _load_hotel_names() -> list[str]:
    global _hotel_names
    if _hotel_names is not None:
        return _hotel_names
    path = os.path.join(OUTPUT_DIR, "hotel_names.json")
    if not os.path.isfile(path):
        return []
    with open(path, encoding="utf-8") as f:
        _hotel_names = json.load(f)
    return _hotel_names


def _resolve_hotel(name: str) -> tuple[str, float, bool]:
    """
    Fuzzy-match name against the known hotel list.
    Returns (resolved_name, confidence_score, unresolved_flag).
    """
    if not name or name == "__global__":
        return "__global__", 100.0, False

    try:
        from rapidfuzz import fuzz, process as rfprocess
        hotel_names = _load_hotel_names()
        if not hotel_names:
            return name, 100.0, False

        result = rfprocess.extractOne(name, hotel_names, scorer=fuzz.ratio)
        if result and result[1] >= HOTEL_RESOLUTION_THRESHOLD:
            return result[0], float(result[1]), False
        score = float(result[1]) if result else 0.0
        return name, score, True

    except ImportError:
        return name, 100.0, False


# ── LLM ───────────────────────────────────────────────────────────────────────

_llm: ChatOpenAI | None = None


def _get_llm() -> ChatOpenAI:
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(model="gpt-4o", temperature=0)
    return _llm


# ── Node ──────────────────────────────────────────────────────────────────────

def query_classifier(state: AgentState) -> dict:
    chain = CLASSIFIER_PROMPT | _get_llm()
    history_str = _format_history(state.get("conversation_history", []))

    try:
        result = chain.invoke({
            "query":                state["query"],
            "hotel_name":           state.get("hotel_name", "__global__"),
            "conversation_history": history_str,
        })
        raw = result.content.strip()

        # Strip markdown fences if model wrapped the JSON
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        parsed = ClassifierOutput(**json.loads(raw))

    except Exception as e:
        print(f"[query_classifier] parse error ({e}) — using fallback")
        parsed = ClassifierOutput(hotel_name=state.get("hotel_name", "__global__"))

    resolved_name, confidence, unresolved = _resolve_hotel(parsed.hotel_name)

    return {
        "query_type":       parsed.query_type,
        "query_direction":  parsed.query_direction,
        "aspects":          parsed.aspects,
        "segment":          parsed.segment,
        "hotel_name":       resolved_name,
        "hotel_confidence": confidence,
        "hotel_unresolved": unresolved,
        "last_topic":       state.get("query_type"),
    }


def _format_history(history: list[dict]) -> str:
    if not history:
        return "None"
    lines = []
    for turn in history[-4:]:
        lines.append(f"User: {turn.get('query', '')}")
        lines.append(f"Assistant: {turn.get('response', '')[:200]}")
    return "\n".join(lines)
