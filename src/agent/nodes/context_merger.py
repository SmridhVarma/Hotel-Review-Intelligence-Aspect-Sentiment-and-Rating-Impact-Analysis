# context_merger.py — Stage 5 | Module C (Agent): Merges retrieved chunks and SHAP summary into a formatted context string; sets low_confidence on hard failures.
#
# Input:  AgentState — retrieved_chunks, summary_context, hotel_unresolved, insufficient_data
# Output: AgentState — low_confidence

from __future__ import annotations

from agent.state import AgentState


def context_merger(state: AgentState) -> dict:
    chunks           = state.get("retrieved_chunks", [])
    summary_context  = state.get("summary_context")
    hotel_unresolved = state.get("hotel_unresolved", False)

    # Hard failure: hotel name is ambiguous — can't retrieve anything meaningful
    if hotel_unresolved:
        return {"low_confidence": True}

    # Hard failure: no data at all
    if not chunks and not summary_context:
        return {"low_confidence": True}

    # Everything else → best-guess answer using whatever was retrieved
    return {"low_confidence": False}
