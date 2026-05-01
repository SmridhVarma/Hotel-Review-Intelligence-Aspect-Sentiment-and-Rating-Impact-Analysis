# state_manager.py — Stage 5 | Module C (Agent): Last DAG node; appends the turn to conversation history and clears retrieval cache on topic shift.
#
# Input:  AgentState — query, response, query_type, last_topic, hyde_embeddings, hyde_hypotheticals, retrieved_chunks
# Output: AgentState — conversation_history (appended), last_topic, hyde_embeddings/hypotheticals/retrieved_chunks (cleared on topic shift)

from __future__ import annotations

from agent.state import AgentState


def state_manager(state: AgentState) -> dict:
    current_topic  = state.get("query_type", "evidence")
    previous_topic = state.get("last_topic")
    topic_shift    = previous_topic is not None and previous_topic != current_topic

    new_turn = {
        "query":    state.get("query", ""),
        "response": state.get("response", ""),
        "hotel":    state.get("hotel_name", "__global__"),
        "topic":    current_topic,
    }

    update: dict = {
        "conversation_history": [new_turn],   # operator.add appends this
        "last_topic":           current_topic,
    }

    if topic_shift:
        update["hyde_hypotheticals"] = []
        update["hyde_embeddings"]    = []
        update["retrieved_chunks"]   = []

    return update
