# =============================================================================
# app.py — Gradio Chatbot Web Interface
# =============================================================================
# Purpose:
#   Launches the Gradio web UI for the hotel review intelligence chatbot.
#   Hotel selector drives the hotel_name field in AgentState — all retrieval
#   scopes automatically to the selected hotel.
#
# Runtime dependencies:
#   src/agent/graph.py must be importable
#   chromadb/ must be populated (run scripts/run_pipeline.py first)
#   outputs/shap_summary.json must exist (hotel name list)
#   OPENAI_API_KEY in .env
#
# Tech stack:
#   gradio, src.agent.graph (compiled StateGraph)
#
# =============================================================================
# HOW TO CALL THE AGENT
# =============================================================================
#
# from src.agent.graph import app as agent_app
#
# Each invocation needs an initial state dict and a config with thread_id.
# The thread_id is what LangGraph's MemorySaver uses to persist state across
# turns — every Gradio session should get a unique thread_id (e.g. uuid4).
# Two users with different thread_ids have fully isolated conversation state.
#
# Minimal state to pass per query:
#
#   state = {
#       "query":                str,   # the user's message
#       "hotel_name":           str,   # "__global__" or specific hotel name
#       "conversation_history": list,  # list of {"role": "user"|"assistant",
#                                      #           "content": str}
#       "session_id":           str,   # same as thread_id below
#   }
#
#   config = {"configurable": {"thread_id": session_id}}
#   result = agent_app.invoke(state, config=config)
#
# All other AgentState fields (scope, query_type, aspects, hypotheticals, etc.)
# are filled in by the graph nodes — do not set them manually.
#
# On the first turn, pass conversation_history=[].
# On subsequent turns, pass the full history so the query_classifier can
# resolve follow-up references like "what about for business travelers?"
# without the user repeating the hotel name.
#
# =============================================================================
# WHAT TO READ FROM THE RESULT
# =============================================================================
#
#   result["answer"]           str        — response text, display in chat
#   result["citations"]        list[dict] — sources to show below the response
#                                           each dict has:
#                                             quote            (str)
#                                             aspect           (str)
#                                             sentiment        (str)
#                                             reviewer_segment (str)
#                                             reviewer_score   (float)
#                                             similarity       (float)
#   result["hotel_unresolved"] bool       — True if the hotel name could not be
#                                           matched with confidence > 70.
#                                           Show a warning: "Could not confidently
#                                           identify hotel — showing best guess."
#   result["insufficient_data"] bool      — True if hotel has < 100 reviews.
#                                           Show a notice: "Limited review data
#                                           for this hotel — rankings may vary."
#
# =============================================================================
# SUGGESTED UI COMPONENTS
# =============================================================================
#
#   Hotel dropdown:
#     Options: "All Hotels" (maps to "__global__") + sorted hotel names
#     Source: outputs/shap_summary.json — load hotel_name for each entry
#             where hotel_name != "__global__"
#     On change: reset conversation_history and generate a new session_id
#                so memory doesn't carry over to a different hotel context
#
#   Chat panel:
#     Standard multi-turn Gradio ChatInterface
#     Pass conversation_history as the accumulated turn log each time
#
#   Citations panel (below each response):
#     Show top 5 citations from result["citations"]
#     Format: [Aspect | Sentiment | Segment | Score: X.X]
#             "quote text..."
#
#   Inline warnings (above the response if flags are set):
#     hotel_unresolved=True  → yellow notice about hotel name confidence
#     insufficient_data=True → grey notice about limited review data
#
# =============================================================================
