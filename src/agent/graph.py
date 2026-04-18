# =============================================================================
# graph.py — LangGraph StateGraph Wiring
# =============================================================================
# Purpose:
#   Assembles the LangGraph StateGraph by importing node functions from
#   nodes/ and defining the edges between them. This file contains no
#   business logic — it is purely wiring.
#
#   Node execution order:
#     query_classifier
#       → segment_filter
#         → [conditional]
#             prioritization/mismatch → summary_retriever → context_merger
#             evidence/segment        → hyde_expander → evidence_retriever → context_merger
#         → context_merger
#       → response_generator
#       → state_manager → END
#
# Input:
#   query                (str)       current user message
#   hotel_name           (str)       "__global__" or specific hotel
#   conversation_history (list[dict]) prior turns
#
# Output:
#   response   (str)        agent response text
#   citations  (list[dict]) source metadata for UI display
#
# Tech stack (locked):
#   langgraph (StateGraph, END), src.agent.state (ConversationState),
#   all nodes in src.agent.nodes.*
# =============================================================================
