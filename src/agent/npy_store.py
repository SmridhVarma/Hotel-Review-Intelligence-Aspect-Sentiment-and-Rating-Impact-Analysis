"""
src/agent/npy_store.py — Numpy-backed vector store (chromadb drop-in).

Used as a fallback when chromadb is not installed (e.g. HF Spaces where
chromadb's Rust/C++ dependencies time out during compilation).

Provides a minimal subset of the chromadb API used by evidence_retriever
and summary_retriever:

    client = NpyClient(path)
    col    = client.get_collection("evidence_store")
    col.query(query_embeddings=[...], n_results=20, where={...}, include=[...])
    col.get(where={...}, include=[...])

Data files (written by scripts/export_demo_vectors.py):
    {path}/evidence_store_embeddings.npy   # (N, 1536) float32
    {path}/evidence_store_meta.json        # [{id, document, metadata}, ...]
    {path}/summary_store_embeddings.npy
    {path}/summary_store_meta.json
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np


# ── Where filter evaluator ─────────────────────────────────────────────────────

def _matches(meta: dict, where: dict) -> bool:
    """Evaluate a ChromaDB-style where filter against a metadata dict."""
    if "$and" in where:
        return all(_matches(meta, cond) for cond in where["$and"])
    for field, condition in where.items():
        if isinstance(condition, dict):
            if "$eq" in condition:
                if meta.get(field) != condition["$eq"]:
                    return False
        else:
            if meta.get(field) != condition:
                return False
    return True


# ── Collection ─────────────────────────────────────────────────────────────────

class NpyCollection:
    """Lazy-loading collection backed by .npy + .json files."""

    def __init__(self, vectors_dir: str, name: str) -> None:
        self._vectors_dir = vectors_dir
        self._name = name
        self._embeddings: np.ndarray | None = None   # (N, D) float32
        self._ids:        list[str]          | None = None
        self._docs:       list[str]          | None = None
        self._metas:      list[dict]         | None = None

    def _load(self) -> None:
        if self._embeddings is not None:
            return
        emb_path  = os.path.join(self._vectors_dir, f"{self._name}_embeddings.npy")
        meta_path = os.path.join(self._vectors_dir, f"{self._name}_meta.json")
        if not os.path.isfile(emb_path) or not os.path.isfile(meta_path):
            raise FileNotFoundError(
                f"Demo vectors not found at {self._vectors_dir}/\n"
                "Run: python scripts/export_demo_vectors.py"
            )
        self._embeddings = np.load(emb_path)                       # (N, D)
        with open(meta_path, encoding="utf-8") as f:
            records = json.load(f)
        self._ids   = [r["id"]       for r in records]
        self._docs  = [r["document"] for r in records]
        self._metas = [r["metadata"] for r in records]

    def _filter_indices(self, where: dict | None) -> np.ndarray:
        """Return integer indices of rows matching the where filter."""
        if not where:
            return np.arange(len(self._ids))
        mask = np.array([_matches(m, where) for m in self._metas], dtype=bool)
        return np.where(mask)[0]

    def query(
        self,
        query_embeddings: list[list[float]],
        n_results: int = 10,
        where: dict | None = None,
        include: list[str] | None = None,
        **_: Any,
    ) -> dict:
        """
        chromadb-compatible similarity query.
        Returns nested lists (one per query embedding).
        """
        self._load()
        q = np.array(query_embeddings[0], dtype=np.float32)

        # Candidate set (filtered)
        idx = self._filter_indices(where)
        if len(idx) == 0:
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

        # Normalised cosine similarity
        embs = self._embeddings[idx]                                       # (K, D)
        q_norm    = q    / (np.linalg.norm(q)                + 1e-10)
        emb_norms = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-10)
        sims = emb_norms @ q_norm                                          # (K,)

        # Top-k
        k = min(n_results, len(idx))
        top_local  = np.argsort(sims)[::-1][:k]
        top_global = idx[top_local]

        ids       = [self._ids[i]   for i in top_global]
        docs      = [self._docs[i]  for i in top_global]
        metas     = [self._metas[i] for i in top_global]
        distances = [float(1 - sims[j]) for j in top_local]   # cosine distance

        return {
            "ids":       [ids],
            "documents": [docs],
            "metadatas": [metas],
            "distances": [distances],
        }

    def get(
        self,
        ids:     list[str] | None = None,
        where:   dict      | None = None,
        include: list[str] | None = None,
        **_: Any,
    ) -> dict:
        """
        chromadb-compatible get (flat lists, no similarity).
        Supports lookup by ids or by where filter.
        """
        self._load()
        if ids is not None:
            id_set  = set(ids)
            indices = [i for i, id_ in enumerate(self._ids) if id_ in id_set]
        elif where is not None:
            indices = list(self._filter_indices(where))
        else:
            indices = list(range(len(self._ids)))

        return {
            "ids":       [self._ids[i]   for i in indices],
            "documents": [self._docs[i]  for i in indices],
            "metadatas": [self._metas[i] for i in indices],
        }


# ── Client ─────────────────────────────────────────────────────────────────────

class NpyClient:
    """Drop-in for chromadb.PersistentClient backed by numpy files."""

    def __init__(self, path: str) -> None:
        self._path = path
        self._cache: dict[str, NpyCollection] = {}

    def get_collection(self, name: str) -> NpyCollection:
        if name not in self._cache:
            self._cache[name] = NpyCollection(self._path, name)
        return self._cache[name]
