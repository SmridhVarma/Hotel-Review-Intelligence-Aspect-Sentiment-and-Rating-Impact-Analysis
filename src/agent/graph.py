# graph.py — Stage 5 | Module C (Agent): Wires the LangGraph StateGraph DAG; defines node connections and conditional routing. No business logic.
#
# Input:  None (imported by app.py and src/ui/app.py)
# Output: None

from __future__ import annotations

import os
import sys

_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from agent.nodes.context_merger import context_merger
from agent.nodes.evidence_retriever import evidence_retriever
from agent.nodes.hyde_expander import hyde_expander
from agent.nodes.query_classifier import query_classifier
from agent.nodes.response_generator import response_generator
from agent.nodes.segment_filter import segment_filter
from agent.nodes.state_manager import state_manager
from agent.nodes.summary_retriever import summary_retriever
from agent.state import AgentState


# ── Routing function ──────────────────────────────────────────────────────────

def _route_after_segment_filter(state: AgentState) -> str:
    """
    After segment_filter, decide retrieval path based on query_type.
    Prioritization and mismatch go to summary_store.
    Evidence queries go through HyDE → evidence_store.
    """
    if state.get("query_type") in ("prioritization", "mismatch"):
        return "summary_retriever"
    return "hyde_expander"


# ── Graph assembly ────────────────────────────────────────────────────────────

def build_graph():
    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node("query_classifier",  query_classifier)
    graph.add_node("segment_filter",    segment_filter)
    graph.add_node("hyde_expander",     hyde_expander)
    graph.add_node("evidence_retriever", evidence_retriever)
    graph.add_node("summary_retriever", summary_retriever)
    graph.add_node("context_merger",    context_merger)
    graph.add_node("response_generator", response_generator)
    graph.add_node("state_manager",     state_manager)

    # Entry point
    graph.set_entry_point("query_classifier")

    # Linear edges
    graph.add_edge("query_classifier",   "segment_filter")
    graph.add_edge("hyde_expander",      "evidence_retriever")
    graph.add_edge("evidence_retriever", "context_merger")
    graph.add_edge("summary_retriever",  "context_merger")
    graph.add_edge("context_merger",     "response_generator")
    graph.add_edge("response_generator", "state_manager")
    graph.add_edge("state_manager",      END)

    # Conditional routing after segment_filter
    graph.add_conditional_edges(
        "segment_filter",
        _route_after_segment_filter,
        {
            "hyde_expander":     "hyde_expander",
            "summary_retriever": "summary_retriever",
        },
    )

    memory = MemorySaver()
    return graph.compile(checkpointer=memory)


# ── Compiled app — import this in ui/app.py ───────────────────────────────────

app = build_graph()


# ── Helper for single-turn invocation ─────────────────────────────────────────

def run_query(
    query: str,
    hotel_name: str = "__global__",
    thread_id: str = "default",
) -> dict:
    """
    Convenience wrapper for invoking the graph with a single query.

    Returns the full output state dict including response and citations.
    thread_id identifies the conversation session for MemorySaver.
    """
    from agent.state import DEFAULT_STATE

    initial = {
        **DEFAULT_STATE,
        "query":      query,
        "hotel_name": hotel_name,
    }

    config = {"configurable": {"thread_id": thread_id}}
    result = app.invoke(initial, config=config)
    return result
