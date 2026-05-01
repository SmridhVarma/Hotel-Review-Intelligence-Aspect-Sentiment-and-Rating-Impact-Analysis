# response_generator.py — Stage 5 | Module C (Agent): Calls GPT-4o to generate the final response with citations; falls back to a cannot-answer message on low confidence.
#
# Input:  AgentState — query, hotel_name, retrieved_chunks, summary_context, low_confidence, hotel_unresolved, insufficient_data
# Output: AgentState — response, citations

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

_SRC = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from agent.prompts import CANNOT_ANSWER_PROMPT, RESPONSE_PROMPT
from agent.state import AgentState

load_dotenv()

MAX_CITATIONS = 5   # citations shown in the UI panel

_llm: ChatOpenAI | None = None


def _get_llm() -> ChatOpenAI:
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(model="gpt-4o", temperature=0)
    return _llm


def _format_retrieved_context(chunks: list[dict], summary: str | None) -> str:
    """Format chunks and summary into a numbered source list for the prompt."""
    lines = []

    if summary:
        lines.append("SHAP Impact Summary:")
        lines.append(summary)
        lines.append("")

    if chunks:
        lines.append("Review evidence:")
        for i, chunk in enumerate(chunks, 1):
            seg   = chunk.get("reviewer_segment", "")
            score = chunk.get("reviewer_score", "")
            asp   = chunk.get("aspect", "")
            sent  = chunk.get("sentiment", "")
            lines.append(
                f"[Source {i}] ({asp}, {sent}, {seg}, score {score}): "
                f"{chunk['text']}"
            )

    return "\n".join(lines)


def _low_confidence_reason(state: AgentState) -> str:
    if state.get("hotel_unresolved"):
        return (
            f"The hotel name could not be resolved with confidence "
            f"(match score: {state.get('hotel_confidence', 0):.0f}/100)."
        )
    if state.get("insufficient_data"):
        return "This hotel has too few reviews for reliable SHAP analysis."
    return "The retrieved review sentences were not sufficiently relevant to the query."


def _format_citations(chunks: list[dict]) -> list[dict]:
    return [
        {
            "text":             c["text"],
            "hotel_name":       c.get("hotel_name", ""),
            "aspect":           c.get("aspect", ""),
            "sentiment":        c.get("sentiment", ""),
            "reviewer_segment": c.get("reviewer_segment", ""),
            "reviewer_score":   c.get("reviewer_score", 0.0),
            "similarity_score": c.get("similarity_score", 0.0),
        }
        for c in chunks[:MAX_CITATIONS]
    ]


def response_generator(state: AgentState) -> dict:
    llm            = _get_llm()
    chunks         = state.get("retrieved_chunks", [])
    summary        = state.get("summary_context")
    hotel_name     = state.get("hotel_name", "__global__")
    hotel_context  = hotel_name if hotel_name != "__global__" else "all hotels"
    history_str    = _format_history(state.get("conversation_history", []))

    if state.get("low_confidence"):
        chain = CANNOT_ANSWER_PROMPT | llm
        result = chain.invoke({
            "query":         state["query"],
            "hotel_context": hotel_context,
            "reason":        _low_confidence_reason(state),
        })
        return {"response": result.content.strip(), "citations": []}

    context_str = _format_retrieved_context(chunks, summary)
    chain = RESPONSE_PROMPT | llm
    result = chain.invoke({
        "query":              state["query"],
        "hotel_context":      hotel_context,
        "retrieved_context":  context_str,
        "conversation_history": history_str,
    })

    return {
        "response":  result.content.strip(),
        "citations": _format_citations(chunks),
    }


def _format_history(history: list[dict]) -> str:
    if not history:
        return "None"
    lines = []
    for turn in history[-4:]:
        lines.append(f"User: {turn.get('query', '')}")
        lines.append(f"Assistant: {turn.get('response', '')[:300]}")
    return "\n".join(lines)
