# summary_retriever.py — Stage 5 | Module C (Agent): Fetches the SHAP impact narrative for the selected hotel from ChromaDB summary_store.
#
# Input:  AgentState — hotel_name
# Output: AgentState — summary_context, insufficient_data

from __future__ import annotations

import os
import sys

_SRC = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from agent.state import AgentState
from paths import CHROMADB_DIR, DEMO_VECTORS_DIR

CHROMA_SUMMARY = "summary_store"

_chroma_client = None
_collection    = None


def _get_collection():
    global _chroma_client, _collection
    if _collection is not None:
        return _collection
    try:
        import chromadb
        _chroma_client = chromadb.PersistentClient(path=str(CHROMADB_DIR))
        _collection = _chroma_client.get_collection(CHROMA_SUMMARY)
    except (ImportError, Exception):
        from agent.npy_store import NpyClient
        _chroma_client = NpyClient(path=str(DEMO_VECTORS_DIR))
        _collection = _chroma_client.get_collection(CHROMA_SUMMARY)
    return _collection


def summary_retriever(state: AgentState) -> dict:
    hotel_name = state.get("hotel_name", "__global__")
    collection = _get_collection()

    results = collection.get(
        where={"hotel_name": {"$eq": hotel_name}},
        include=["documents", "metadatas"],
    )

    if not results["documents"]:
        # Fall back to global if hotel-specific entry not found
        if hotel_name != "__global__":
            results = collection.get(
                where={"hotel_name": {"$eq": "__global__"}},
                include=["documents", "metadatas"],
            )

    if not results["documents"]:
        return {"summary_context": None, "insufficient_data": False}

    doc  = results["documents"][0]
    meta = results["metadatas"][0]
    insufficient = bool(meta.get("insufficient_data", False))

    return {
        "summary_context":  doc,
        "insufficient_data": insufficient,
    }
