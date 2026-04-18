# =============================================================================
# app.py — Gradio Chatbot Web Interface
# =============================================================================
# Purpose:
#   Launches the Gradio web UI for the hotel review intelligence chatbot.
#   Hotel selector drives the hotel_name field in ConversationState —
#   all retrieval scopes automatically to the selected hotel.
#
# UI Components:
#   Hotel dropdown   "All Hotels" (default, maps to "__global__") +
#                    1,492 hotel names loaded from outputs/shap_summary.json
#   Chat interface   multi-turn, passes conversation_history each turn
#   Citations panel  displays retrieved_chunks metadata below each response
#
# Input:
#   User messages      via Gradio chat
#   Selected hotel     via Gradio dropdown → hotel_name in state
#
# Output:
#   Chat responses + citations at http://localhost:7860
#
# Runtime dependencies:
#   src/agent/graph.py must be importable
#   chromadb/ must be populated (run scripts/run_pipeline.py first)
#   outputs/shap_summary.json must exist (hotel name list)
#   OPENAI_API_KEY in .env
#
# Tech stack (locked):
#   gradio, src.agent.graph (compiled StateGraph)
# =============================================================================
