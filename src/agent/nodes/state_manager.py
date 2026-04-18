# =============================================================================
# state_manager.py — Node: Conversation State Persistence
# =============================================================================
# Purpose:
#   Last node in the DAG before END. Appends the current turn to
#   conversation_history and updates last_topic. If query_type has changed
#   from last_topic (topic shift detected), clears hyde_embedding and
#   retrieved_chunks to force fresh retrieval on the next turn.
#
# Reads from state:
#   query                (str)        current user message
#   response             (str)        generated response
#   query_type           (str)        current query classification
#   last_topic           (str | None) previous query_type
#   hyde_embedding       (list[float] | None)
#   retrieved_chunks     (list[dict])
#   conversation_history (list[dict])
#
# Writes to state:
#   conversation_history  (list[dict])  appended with current turn
#   last_topic            (str)         set to current query_type
#   hyde_embedding        (None)        cleared on topic shift
#   retrieved_chunks      ([])          cleared on topic shift
# =============================================================================
