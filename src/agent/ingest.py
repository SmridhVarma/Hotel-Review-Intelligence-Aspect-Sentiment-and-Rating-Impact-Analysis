"""
Stage 5: ChromaDB Ingestion

Embeds all review sentences and SHAP summaries into a local ChromaDB instance
for retrieval by the LangGraph agent. Run once after the full pipeline completes.
Resume-safe — skips IDs already present in each collection.

Usage:
    python -m src.agent.ingest

Collections:
    evidence_store  — ~735k sentence-level documents
                      metadata: hotel_name, aspect, sentiment, reviewer_segment,
                                reviewer_score
                      used by: evidence_retriever node

    summary_store   — ~1,493 SHAP narrative documents (one per hotel + __global__)
                      metadata: hotel_name, review_count, insufficient_data
                      used by: summary_retriever node

Cost estimate:
    ~735k sentences × ~20 tokens avg = ~14.7M tokens
    text-embedding-3-small at $0.02/1M tokens ≈ $0.30
    Runtime: ~20–40 min depending on OpenAI rate limits.
"""

from __future__ import annotations

import json
import os
import sys
import time

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from paths import ASPECT_SENTENCES, SHAP_SUMMARY, CHROMADB_DIR

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

EMBED_MODEL = "text-embedding-3-small"
EMBED_BATCH = 500          # rows per OpenAI embeddings API call
BATCH_SLEEP = 0.5          # seconds between batches (rate limit headroom)
CHUNK_SIZE  = 2500         # embed + upsert this many rows at a time (saves progress)
MIN_REVIEWS = 20           # hotels below this threshold get insufficient_data=True
CHROMA_EVIDENCE = "evidence_store"
CHROMA_SUMMARY  = "summary_store"


# ── Clients (initialised at call time, not module import) ─────────────────────

def get_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY not set. Add it to .env")
    return OpenAI(api_key=api_key)


def get_chroma_client():
    import chromadb
    os.makedirs(CHROMADB_DIR, exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMADB_DIR))


# ── Embedding helper ──────────────────────────────────────────────────────────

def _embed_batch_with_retry(batch: list[str], client: OpenAI, max_retries: int = 12) -> list[list[float]]:
    """Embed one batch with exponential-backoff retry on transient errors."""
    delay = 10.0
    for attempt in range(max_retries):
        try:
            response = client.embeddings.create(
                model=EMBED_MODEL, input=batch, timeout=120
            )
            return [e.embedding for e in response.data]
        except Exception as exc:
            if attempt == max_retries - 1:
                raise
            print(f"\n  Embedding error (attempt {attempt + 1}/{max_retries}): {exc}")
            print(f"  Retrying in {delay:.0f}s...")
            time.sleep(delay)
            delay = min(delay * 2, 180)
    return []  # unreachable


def embed_texts(texts: list[str], client: OpenAI) -> list[list[float]]:
    """
    Embed a list of texts in batches using text-embedding-3-small.
    Returns a flat list of 1536-dim vectors in the same order as input.
    Retries each batch up to 5 times with exponential backoff.
    """
    all_embeddings: list[list[float]] = []
    total = len(texts)

    for start in range(0, total, EMBED_BATCH):
        batch = texts[start : start + EMBED_BATCH]
        # Replace empty strings — OpenAI rejects them
        batch = [t if t.strip() else "." for t in batch]

        batch_embeddings = _embed_batch_with_retry(batch, client)
        all_embeddings.extend(batch_embeddings)

        done = min(start + EMBED_BATCH, total)
        print(f"  Embedded {done:,} / {total:,}", end="\r")

        if done < total:
            time.sleep(BATCH_SLEEP)

    print(f"  Embedded {total:,} / {total:,} — done.          ")
    return all_embeddings


# ── SHAP narrative formatter ───────────────────────────────────────────────────

def format_shap_narrative(entry: dict) -> str:
    """
    Convert a shap_summary.json entry into a human-readable string for embedding.

    Input:  { "hotel_name": "Hotel Arena",
              "aspect_impacts": { "cleanliness": -0.42, "staff": 0.31, ... } }
    Output: "Hotel Arena. Aspect impact ranking based on global model attribution:
             Staff: +0.31, Location: +0.28, Room: -0.08, Noise: -0.19,
             Cleanliness: -0.42, Food: +0.11."
    """
    name = entry["hotel_name"]
    impacts: dict[str, float] = entry["aspect_impacts"]

    # Sort by absolute impact descending
    ranked = sorted(impacts.items(), key=lambda x: abs(x[1]), reverse=True)
    impact_str = ", ".join(
        f"{asp.capitalize()}: {val:+.2f}" for asp, val in ranked
    )

    positive = [asp.capitalize() for asp, val in ranked if val > 0]
    negative = [asp.capitalize() for asp, val in ranked if val < 0]

    pos_str = ", ".join(positive) if positive else "none"
    neg_str = ", ".join(negative) if negative else "none"

    label = "Global (all hotels)" if name == "__global__" else name
    return (
        f"{label}. Aspect impact ranking based on global model attribution: "
        f"{impact_str}. "
        f"Aspects with positive impact: {pos_str}. "
        f"Aspects with negative impact: {neg_str}."
    )


# ── Evidence store ingestion ──────────────────────────────────────────────────

def ingest_evidence_store(
    df: pd.DataFrame,
    chroma_client,
    openai_client: OpenAI,
) -> None:
    """
    Embed all rows in aspect_sentences.csv and upsert into evidence_store.
    Resume-safe: embeds + upserts in small chunks so progress is saved
    incrementally. On restart, skips the chunks already stored.

    Each document is the sentence text. Metadata fields:
        hotel_name, aspect, sentiment, reviewer_segment, reviewer_score
    """
    total = len(df)
    print(f"\nIngesting evidence_store ({total:,} sentences)...")

    # Get or create collection (no delete — preserves progress on resume)
    try:
        collection = chroma_client.get_collection(CHROMA_EVIDENCE)
        already_stored = collection.count()
        print(f"  Resuming — {already_stored:,} documents already stored.")
    except Exception:
        collection = chroma_client.create_collection(
            name=CHROMA_EVIDENCE,
            metadata={"hnsw:space": "cosine"},
        )
        already_stored = 0

    if already_stored >= total:
        print(f"  evidence_store already complete ({already_stored:,} documents).")
        return

    # IDs are sequential (ev_0, ev_1, ...) so we can use count to find the
    # resume offset.  This avoids fetching all 735k IDs into a set.
    start_idx = already_stored
    remaining = total - start_idx
    print(f"  {remaining:,} sentences remaining (starting from index {start_idx:,})...")

    texts     = df["sentence"].astype(str).tolist()
    metadatas = [
        {
            "hotel_name":       str(row.hotel_name),
            "aspect":           str(row.aspect),
            "sentiment":        str(row.sentiment),
            "reviewer_segment": str(row.reviewer_segment),
            "reviewer_score":   float(row.reviewer_score),
        }
        for row in df.itertuples(index=False)
    ]

    # Process in small chunks: embed -> upsert -> repeat.
    # Each chunk is saved immediately, so a crash only loses the current chunk.
    for chunk_start in range(start_idx, total, CHUNK_SIZE):
        chunk_end = min(chunk_start + CHUNK_SIZE, total)
        chunk_ids   = [f"ev_{i}" for i in range(chunk_start, chunk_end)]
        chunk_texts = texts[chunk_start:chunk_end]
        chunk_meta  = metadatas[chunk_start:chunk_end]

        # Replace empty strings — OpenAI rejects them
        chunk_texts_clean = [t if t.strip() else "." for t in chunk_texts]

        print(f"\n  Chunk [{chunk_start:,}-{chunk_end:,}] / {total:,}  "
              f"Embedding {len(chunk_texts_clean)} texts...")

        # Embed this chunk
        chunk_embeddings = embed_texts(chunk_texts_clean, openai_client)

        # Upsert immediately — this saves progress
        collection.upsert(
            ids=chunk_ids,
            documents=chunk_texts,
            embeddings=chunk_embeddings,
            metadatas=chunk_meta,
        )
        stored = collection.count()
        print(f"  [OK] Saved. Total stored: {stored:,} / {total:,}")

    print(f"\n  evidence_store: {collection.count():,} documents -- done.")


# ── Summary store ingestion ───────────────────────────────────────────────────

def ingest_summary_store(
    shap_data: list[dict],
    chroma_client,
    openai_client: OpenAI,
) -> None:
    """
    Format SHAP narratives and upsert into summary_store.

    Each document is a human-readable SHAP narrative string.
    Metadata: hotel_name, review_count, insufficient_data.

    review_count is computed from the shap_data entries.
    The __global__ entry always has insufficient_data=False.
    """
    print(f"\nIngesting summary_store ({len(shap_data):,} entries)...")

    # Get or create collection — resume-safe
    try:
        collection = chroma_client.get_collection(CHROMA_SUMMARY)
        existing = set(collection.get(include=[])["ids"])
        print(f"  Resuming — {len(existing):,} IDs already stored.")
    except Exception:
        collection = chroma_client.create_collection(
            name=CHROMA_SUMMARY,
            metadata={"hnsw:space": "cosine"},
        )
        existing = set()

    texts     = []
    ids       = []
    metadatas = []

    for entry in shap_data:
        hotel_name   = entry["hotel_name"]
        id_          = f"sum_{hotel_name}"
        if id_ in existing:
            continue
        review_count = entry.get("review_count", 0)
        narrative    = format_shap_narrative(entry)
        texts.append(narrative)
        ids.append(id_)
        insuf = entry.get(
            "insufficient_data",
            hotel_name != "__global__" and review_count < MIN_REVIEWS,
        )
        metadatas.append({
            "hotel_name":        hotel_name,
            "review_count":      int(review_count),
            "insufficient_data": bool(insuf),
        })

    if not texts:
        print(f"  summary_store already complete ({collection.count():,} documents).")
        return

    print(f"  {len(texts):,} entries to embed...")
    embeddings = embed_texts(texts, openai_client)

    collection.upsert(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    print(f"  summary_store: {collection.count():,} documents.")


# ── Hotel name list export ────────────────────────────────────────────────────

def export_hotel_list(df: pd.DataFrame) -> None:
    """
    Write a sorted list of unique hotel names to outputs/hotel_names.json.
    Used by query_classifier for fuzzy hotel name resolution.
    """
    import json
    from paths import OUTPUT_DIR

    hotel_names = sorted(df["hotel_name"].dropna().unique().tolist())
    out_path = os.path.join(OUTPUT_DIR, "hotel_names.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(hotel_names, f, indent=2)
    print(f"\n  Exported {len(hotel_names):,} hotel names -> {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def run() -> None:
    # 1. Check inputs
    if not os.path.isfile(ASPECT_SENTENCES):
        raise FileNotFoundError(
            f"aspect_sentences.csv not found at {ASPECT_SENTENCES}\n"
            "Run Stage 3 (sentiment_assignment.py) first."
        )
    if not os.path.isfile(SHAP_SUMMARY):
        raise FileNotFoundError(
            f"shap_summary.json not found at {SHAP_SUMMARY}\n"
            "Run Stage 4 (model.py) first."
        )

    # 2. Load
    print(f"Loading {ASPECT_SENTENCES}...")
    df = pd.read_csv(ASPECT_SENTENCES, low_memory=False)
    print(f"  {len(df):,} sentences.")

    print(f"Loading {SHAP_SUMMARY}...")
    with open(SHAP_SUMMARY, encoding="utf-8") as f:
        shap_data = json.load(f)

    # Attach review_count to each hotel entry from sentence data
    review_counts = df.groupby("hotel_name")["review_id"].nunique().to_dict()
    for entry in shap_data:
        if entry["hotel_name"] != "__global__":
            entry["review_count"] = review_counts.get(entry["hotel_name"], 0)
        else:
            entry["review_count"] = df["review_id"].nunique()

    # 3. Init clients
    openai_client = get_openai_client()
    chroma_client = get_chroma_client()

    # 4. Ingest
    ingest_evidence_store(df, chroma_client, openai_client)
    ingest_summary_store(shap_data, chroma_client, openai_client)
    export_hotel_list(df)

    # Write sentinel so local_startup.bat knows ingest finished cleanly
    sentinel = os.path.join(CHROMADB_DIR, ".ingest_complete")
    with open(sentinel, "w") as f:
        f.write("ok\n")

    print("\nStage 5 complete.")


if __name__ == "__main__":
    run()


# ── Module-level load guards (for imports without running ingest) ──────────────

df_sentences = pd.read_csv(ASPECT_SENTENCES, low_memory=False) \
    if os.path.isfile(ASPECT_SENTENCES) else None

shap_data = None
if os.path.isfile(SHAP_SUMMARY):
    with open(SHAP_SUMMARY, encoding="utf-8") as f:
        shap_data = json.load(f)
