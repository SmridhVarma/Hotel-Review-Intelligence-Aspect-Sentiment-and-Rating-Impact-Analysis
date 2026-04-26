"""
Node: context_merger

Combines retrieved_chunks and summary_context into a formatted context string
for the response generator. Computes mean cosine similarity across retrieved
chunks to assess retrieval quality.

If mean similarity < CONFIDENCE_THRESHOLD, or both sources returned nothing,
sets low_confidence=True so response_generator uses the fallback prompt.

Reads:  retrieved_chunks, summary_context, hotel_unresolved, insufficient_data
Writes: low_confidence
        (retrieved_chunks and summary_context are read, not modified)
"""

from __future__ import annotations

from agent.state import AgentState

CONFIDENCE_THRESHOLD = 0.50   # mean cosine similarity below this → low confidence


def context_merger(state: AgentState) -> dict:
    chunks          = state.get("retrieved_chunks", [])
    summary_context = state.get("summary_context")
    hotel_unresolved = state.get("hotel_unresolved", False)
    insufficient    = state.get("insufficient_data", False)

    # Hard failure cases
    if hotel_unresolved:
        return {"low_confidence": True}

    has_evidence = bool(chunks)
    has_summary  = bool(summary_context)

    if not has_evidence and not has_summary:
        return {"low_confidence": True}

    # Compute mean similarity from evidence chunks
    if has_evidence:
        scores = [c.get("similarity_score", 0.0) for c in chunks]
        mean_sim = sum(scores) / len(scores)
        low_confidence = mean_sim < CONFIDENCE_THRESHOLD
    else:
        # Summary-only path (prioritization) — trust the lookup
        low_confidence = insufficient

    return {"low_confidence": low_confidence}
