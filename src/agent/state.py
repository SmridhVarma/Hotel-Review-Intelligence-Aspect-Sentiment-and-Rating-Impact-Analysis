"""
Shared conversation state for the LangGraph DAG.

Every node reads from and writes to this TypedDict. Centralising it here
avoids circular imports and duplicated type definitions across node files.

conversation_history uses operator.add so LangGraph appends each turn's
entry rather than overwriting the list.
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict


class AgentState(TypedDict):
    # ── Query ─────────────────────────────────────────────────────────────────
    query: str                           # current user message
    hotel_name: str                      # "__global__" or specific hotel name
    query_type: str                      # "evidence" | "prioritization" | "mismatch"
    query_direction: str                 # "positive" | "negative" | "neutral"
    aspects: list[str]                   # aspects mentioned in query (may be empty)
    segment: str | None                  # "Business"|"Couple"|"Family"|"Solo"|"Group" or None

    # ── Hotel resolution ──────────────────────────────────────────────────────
    hotel_confidence: float              # fuzzy match score 0–100
    hotel_unresolved: bool               # True when confidence < 70
    insufficient_data: bool             # True when hotel has too few reviews for SHAP

    # ── Retrieval ─────────────────────────────────────────────────────────────
    hyde_hypotheticals: list[str]        # generated hypothetical texts (1 or 3)
    hyde_embeddings: list[list[float]]   # embeddings of hypotheticals
    retrieved_chunks: list[dict]         # evidence_store results with metadata
    summary_context: str | None          # summary_store SHAP narrative text
    low_confidence: bool                 # True when mean similarity < threshold

    # ── Output ────────────────────────────────────────────────────────────────
    response: str                        # final agent response text
    citations: list[dict]                # source metadata shown in UI

    # ── Multi-turn ────────────────────────────────────────────────────────────
    conversation_history: Annotated[list[dict], operator.add]
    last_topic: str | None               # previous query_type for topic-shift detection


# Default state — used by graph.py to initialise a fresh conversation turn.
DEFAULT_STATE: AgentState = {
    "query": "",
    "hotel_name": "__global__",
    "query_type": "evidence",
    "query_direction": "neutral",
    "aspects": [],
    "segment": None,
    "hotel_confidence": 100.0,
    "hotel_unresolved": False,
    "insufficient_data": False,
    "hyde_hypotheticals": [],
    "hyde_embeddings": [],
    "retrieved_chunks": [],
    "summary_context": None,
    "low_confidence": False,
    "response": "",
    "citations": [],
    "conversation_history": [],
    "last_topic": None,
}
