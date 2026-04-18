# =============================================================================
# context_merger.py — Node: Context Merging + Confidence Check
# =============================================================================
# Purpose:
#   Combines retrieved_chunks and summary_context into a single context
#   block for the response generator. Computes the mean similarity score
#   across retrieved chunks. If mean score falls below CONFIDENCE_THRESHOLD,
#   sets low_confidence=True so response_generator uses the fallback prompt.
#
#   CONFIDENCE_THRESHOLD is defined as a module-level constant (default 0.5).
#
# Reads from state:
#   retrieved_chunks  (list[dict])  from evidence_retriever (may be empty)
#   summary_context   (str | None)  from summary_retriever (may be None)
#
# Writes to state:
#   low_confidence  (bool)  True if mean similarity < CONFIDENCE_THRESHOLD
#                           or both sources returned nothing
# =============================================================================
