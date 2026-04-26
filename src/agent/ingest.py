"""
Stage 5: ChromaDB Ingestion

Embeds all review sentences and SHAP summaries into a local ChromaDB instance
for retrieval by the LangGraph agent. Run once after the full pipeline completes.
Safe to rerun — clears and rebuilds both collections each time.

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

def embed_texts(texts: list[str], client: OpenAI) -> list[list[float]]:
    """
    Embed a list of texts in batches using text-embedding-3-small.
    Returns a flat list of 1536-dim vectors in the same order as input.
    """
    all_embeddings: list[list[float]] = []
    total = len(texts)

    for start in range(0, total, EMBED_BATCH):
        batch = texts[start : start + EMBED_BATCH]
        # Replace empty strings — OpenAI rejects them
        batch = [t if t.strip() else "." for t in batch]

        response = client.embeddings.create(model=EMBED_MODEL, input=batch)
        batch_embeddings = [e.embedding for e in response.data]
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

    Each document is the sentence text. Metadata fields:
        hotel_name, aspect, sentiment, reviewer_segment, reviewer_score
    """
    print(f"\nIngesting evidence_store ({len(df):,} sentences)...")

    # Drop and recreate for a clean rebuild
    try:
        chroma_client.delete_collection(CHROMA_EVIDENCE)
        print("  Cleared existing evidence_store.")
    except Exception:
        pass

    collection = chroma_client.create_collection(
        name=CHROMA_EVIDENCE,
        metadata={"hnsw:space": "cosine"},
    )

    texts     = df["sentence"].astype(str).tolist()
    ids       = [f"ev_{i}" for i in range(len(df))]
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

    print("  Generating embeddings...")
    embeddings = embed_texts(texts, openai_client)

    # Upsert in batches (ChromaDB handles large upserts fine but batching
    # keeps memory predictable)
    print("  Upserting to ChromaDB...")
    for start in range(0, len(texts), 5000):
        end = min(start + 5000, len(texts))
        collection.upsert(
            ids=ids[start:end],
            documents=texts[start:end],
            embeddings=embeddings[start:end],
            metadatas=metadatas[start:end],
        )
        print(f"  Upserted {end:,} / {len(texts):,}", end="\r")

    print(f"  evidence_store: {collection.count():,} documents.          ")


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

    try:
        chroma_client.delete_collection(CHROMA_SUMMARY)
        print("  Cleared existing summary_store.")
    except Exception:
        pass

    collection = chroma_client.create_collection(
        name=CHROMA_SUMMARY,
        metadata={"hnsw:space": "cosine"},
    )

    texts     = []
    ids       = []
    metadatas = []

    for i, entry in enumerate(shap_data):
        hotel_name   = entry["hotel_name"]
        review_count = entry.get("review_count", 0)
        narrative    = format_shap_narrative(entry)

        texts.append(narrative)
        ids.append(f"sum_{hotel_name}")
        # Prefer the insufficient_data flag set by model.py (threshold=100 reviews)
        # rather than recomputing; fall back to MIN_REVIEWS for entries without it.
        insuf = entry.get(
            "insufficient_data",
            hotel_name != "__global__" and review_count < MIN_REVIEWS,
        )
        metadatas.append({
            "hotel_name":        hotel_name,
            "review_count":      int(review_count),
            "insufficient_data": bool(insuf),
        })

    print("  Generating embeddings...")
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
