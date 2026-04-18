# =============================================================================
# state.py — Shared Conversation State
# =============================================================================
# Purpose:
#   Defines the ConversationState TypedDict that flows through every node
#   in the LangGraph DAG. All nodes read from and write to this state.
#   Centralizing it here means graph.py and every node file imports from
#   one place — no circular imports, no duplicated type definitions.
#
# Used by:
#   graph.py and every file in nodes/
#
# Fields:
#   hotel_name           (str)             selected hotel, "__global__" if none
#   query                (str)             current user message
#   query_type           (str)             "evidence"|"prioritization"|
#                                          "mismatch"|"segment"
#   segment              (str | None)      reviewer segment if detected
#                                          Business|Couple|Family|Solo|Group
#   hyde_embedding       (list[float]|None) embedded hypothetical document
#   retrieved_chunks     (list[dict])      evidence_store results with metadata
#   summary_context      (str | None)      summary_store result text
#   low_confidence       (bool)            True if mean similarity < threshold
#   response             (str)             final agent response text
#   citations            (list[dict])      source metadata shown in UI
#   conversation_history (list[dict])      full turn history for multi-turn context
#   last_topic           (str | None)      previous query_type for topic-shift detection
# =============================================================================
