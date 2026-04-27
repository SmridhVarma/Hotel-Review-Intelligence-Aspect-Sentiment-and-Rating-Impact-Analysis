"""
Node: context_merger

Combines retrieved_chunks and summary_context into a formatted context string
for the response generator.

Sets low_confidence=True only for genuine hard failures:
  - hotel name could not be resolved
  - both evidence and summary are empty (nothing to work with)

Similarity scores are intentionally NOT used as a confidence gate.
text-embedding-3-small scores are ranking metrics, not absolute quality
signals — broad queries naturally score lower, but the retrieved content
is still useful. The response_generator's RESPONSE_PROMPT handles uncertainty
naturally via GPT-4o hedging.

Reads:  retrieved_chunks, summary_context, hotel_unresolved, insufficient_data
Writes: low_confidence
        (retrieved_chunks and summary_context are read, not modified)
"""

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
