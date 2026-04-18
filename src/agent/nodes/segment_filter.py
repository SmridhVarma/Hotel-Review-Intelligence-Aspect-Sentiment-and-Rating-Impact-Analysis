# =============================================================================
# segment_filter.py — Node: Reviewer Segment Detection
# =============================================================================
# Purpose:
#   Extracts reviewer segment from the query (if present) and sets it in
#   state for downstream ChromaDB filtering. Passes through unchanged if
#   no segment is detected — downstream nodes handle None as "no filter".
#
# Reads from state:
#   query        (str)       current user message
#   query_type   (str)       from query_classifier
#
# Writes to state:
#   segment  (str | None)  Business|Couple|Family|Solo|Group or None
# =============================================================================
