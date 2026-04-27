"""
Node: evidence_retriever

Queries ChromaDB evidence_store using the HyDE embeddings from hyde_expander.
Applies optional metadata filters for hotel_name, reviewer_segment, and
(for directional queries) sentiment.

Directional queries (positive/negative): single query, top-20 results,
sentiment filter applied.

Neutral queries: stratified retrieval — one query per hypothetical
(positive / negative / neutral tone), top-7 each, deduplicated by sentence
text. Guarantees the LLM sees a balanced sample of guest experience.

Reads:  hyde_embeddings, hotel_name, segment, query_direction
Writes: retrieved_chunks
"""

from __future__ import annotations

import os
import sys

_SRC = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from agent.state import AgentState
from paths import CHROMADB_DIR, DEMO_VECTORS_DIR

CHROMA_EVIDENCE   = "evidence_store"
DIRECTIONAL_K     = 20    # results for a directional query
STRATIFIED_K      = 7     # results per sentiment pole for neutral queries
SENTIMENT_MAP     = {"positive": "Positive", "negative": "Negative", "neutral": "Neutral"}

_chroma_client = None
_collection    = None


def _get_collection():
    global _chroma_client, _collection
    if _collection is not None:
        return _collection
    try:
        import chromadb
        _chroma_client = chromadb.PersistentClient(path=str(CHROMADB_DIR))
        _collection = _chroma_client.get_collection(CHROMA_EVIDENCE)
    except (ImportError, Exception):
        from agent.npy_store import NpyClient
        _chroma_client = NpyClient(path=str(DEMO_VECTORS_DIR))
        _collection = _chroma_client.get_collection(CHROMA_EVIDENCE)
    return _collection


def _build_where(hotel_name: str, segment: str | None, sentiment: str | None) -> dict | None:
    """Build ChromaDB $and filter. Returns None if no filters apply."""
    conditions = []
    if hotel_name != "__global__":
        conditions.append({"hotel_name": {"$eq": hotel_name}})
    if segment:
        conditions.append({"reviewer_segment": {"$eq": segment}})
    if sentiment:
        conditions.append({"sentiment": {"$eq": sentiment}})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def _query_collection(
    embedding: list[float],
    where: dict | None,
    n_results: int,
) -> list[dict]:
    """Run one ChromaDB similarity query, return list of chunk dicts."""
    collection = _get_collection()
    kwargs = dict(
        query_embeddings=[embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )
    if where:
        kwargs["where"] = where

    results = collection.query(**kwargs)

    chunks = []
    docs      = results["documents"][0]
    metas     = results["metadatas"][0]
    distances = results["distances"][0]

    for doc, meta, dist in zip(docs, metas, distances):
        chunks.append({
            "text":             doc,
            "hotel_name":       meta.get("hotel_name", ""),
            "aspect":           meta.get("aspect", ""),
            "sentiment":        meta.get("sentiment", ""),
            "reviewer_segment": meta.get("reviewer_segment", ""),
            "reviewer_score":   meta.get("reviewer_score", 0.0),
            "similarity_score": round(1 - dist, 4),   # convert distance → similarity
        })
    return chunks


def evidence_retriever(state: AgentState) -> dict:
    embeddings = state.get("hyde_embeddings", [])
    if not embeddings:
        return {"retrieved_chunks": []}

    hotel_name = state.get("hotel_name", "__global__")
    segment    = state.get("segment")
    direction  = state.get("query_direction", "neutral")

    if direction == "neutral" and len(embeddings) >= 3:
        # Stratified retrieval: positive pole, negative pole, unfiltered pole.
        # The data has only Positive/Negative sentiments — the third hypothetical
        # (neutral-tone) queries without a sentiment filter so it can surface any
        # highly-relevant chunk regardless of polarity.
        sentiment_poles = ["Positive", "Negative", None]
        seen_texts: set[str] = set()
        chunks: list[dict] = []

        for embedding, sentiment in zip(embeddings[:3], sentiment_poles):
            where = _build_where(hotel_name, segment, sentiment)
            pole_chunks = _query_collection(embedding, where, STRATIFIED_K)
            for chunk in pole_chunks:
                if chunk["text"] not in seen_texts:
                    seen_texts.add(chunk["text"])
                    chunks.append(chunk)

        # Sort combined results by similarity descending
        chunks.sort(key=lambda c: c["similarity_score"], reverse=True)

    else:
        # Directional: single query with sentiment filter
        chroma_sentiment = SENTIMENT_MAP.get(direction)
        where = _build_where(hotel_name, segment, chroma_sentiment)
        chunks = _query_collection(embeddings[0], where, DIRECTIONAL_K)

    return {"retrieved_chunks": chunks}
