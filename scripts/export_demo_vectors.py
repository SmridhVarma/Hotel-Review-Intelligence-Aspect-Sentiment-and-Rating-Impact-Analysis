"""
scripts/export_demo_vectors.py — Export chromadb_demo to numpy for HF Spaces.

Reads chromadb_demo/ (built by build_demo_db.py) and writes demo_vectors/:
    demo_vectors/
    ├── evidence_embeddings.npy   # (N, 1536) float32  — ~177 MB
    ├── evidence_meta.json        # [{id, document, metadata}, ...]
    ├── summary_embeddings.npy    # (M, 1536) float32
    └── summary_meta.json

These files are used by src/agent/npy_store.py as a chromadb drop-in
that requires no C++/Rust compilation — critical for HF Spaces deployment.

Usage:
    python scripts/export_demo_vectors.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
_ROOT    = os.path.dirname(_SCRIPTS)
_SRC     = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

DEMO_DB_DIR     = os.path.join(_ROOT, "chromadb_demo")
DEMO_VECTORS_DIR = os.path.join(_ROOT, "demo_vectors")
BATCH_SIZE       = 5000


def _get_all(collection, include: list[str]) -> dict:
    """Fetch all docs from a collection in batches (chromadb has no 'get all' shortcut)."""
    total   = collection.count()
    all_ids, all_docs, all_metas, all_embeds = [], [], [], []
    offset  = 0
    while offset < total:
        batch = collection.get(
            limit=BATCH_SIZE,
            offset=offset,
            include=include,
        )
        all_ids.extend(batch["ids"])
        if "documents" in include:
            all_docs.extend(batch["documents"])
        if "metadatas" in include:
            all_metas.extend(batch["metadatas"])
        if "embeddings" in include:
            all_embeds.extend(batch["embeddings"])
        offset += BATCH_SIZE
        print(f"  Fetched {min(offset, total):,} / {total:,}", end="\r")
    print(f"  Fetched {total:,} / {total:,} — done.          ")
    return {
        "ids": all_ids,
        "documents": all_docs,
        "metadatas": all_metas,
        "embeddings": all_embeds,
    }


def export_collection(chroma_client, name: str, out_dir: str) -> None:
    print(f"\nExporting {name} ...")
    collection = chroma_client.get_collection(name)
    data = _get_all(collection, include=["documents", "metadatas", "embeddings"])

    # Numpy embeddings
    emb_path = os.path.join(out_dir, f"{name}_embeddings.npy")
    emb_array = np.array(data["embeddings"], dtype=np.float32)
    np.save(emb_path, emb_array)
    print(f"  Saved {emb_path}  shape={emb_array.shape}  ({emb_array.nbytes / 1e6:.1f} MB)")

    # JSON metadata (ids + documents + metadatas)
    meta_path = os.path.join(out_dir, f"{name}_meta.json")
    meta_records = [
        {"id": id_, "document": doc, "metadata": meta}
        for id_, doc, meta in zip(data["ids"], data["documents"], data["metadatas"])
    ]
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_records, f)
    print(f"  Saved {meta_path}  ({len(meta_records):,} records)")


def main() -> None:
    import chromadb

    if not os.path.isdir(DEMO_DB_DIR):
        raise FileNotFoundError(
            f"chromadb_demo/ not found at {DEMO_DB_DIR}\n"
            "Run: python scripts/build_demo_db.py --hotels 5"
        )

    os.makedirs(DEMO_VECTORS_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=str(DEMO_DB_DIR))

    export_collection(client, "evidence_store", DEMO_VECTORS_DIR)
    export_collection(client, "summary_store",  DEMO_VECTORS_DIR)

    print(f"\nAll done. Files written to {DEMO_VECTORS_DIR}/")


if __name__ == "__main__":
    main()
