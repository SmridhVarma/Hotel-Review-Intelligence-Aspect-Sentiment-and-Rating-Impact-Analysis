# build_demo_db.py — Stage 5 utility | Module C (Agent): Builds a lightweight demo ChromaDB by copying the top N hotels from the full DB (no re-embedding, no API cost).
#
# Input:  chromadb/ (~5.5 GB full DB)
# Output: chromadb_demo/ (lightweight subset, ~300–500 MB)

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
_ROOT    = os.path.dirname(_SCRIPTS)
_SRC     = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from paths import CHROMADB_DIR, OUTPUT_DIR

DEMO_DB_DIR = os.path.join(_ROOT, "chromadb_demo")


def build_demo_db(n_hotels: int = 50) -> None:
    import chromadb

    # ── Source client ────────────────────────────────────────────────────────
    print(f"Opening source ChromaDB at {CHROMADB_DIR} ...")
    if not os.path.isdir(CHROMADB_DIR):
        raise FileNotFoundError(
            f"chromadb/ not found at {CHROMADB_DIR}\n"
            "Download it from OneDrive or run Stage 5 ingestion first."
        )
    src_client = chromadb.PersistentClient(path=str(CHROMADB_DIR))

    # ── Count sentences per hotel ────────────────────────────────────────────
    src_ev    = src_client.get_collection("evidence_store")
    total_docs = src_ev.count()
    print(f"  evidence_store: {total_docs:,} documents total")

    hotel_names_path = os.path.join(OUTPUT_DIR, "hotel_names.json")
    if os.path.isfile(hotel_names_path):
        with open(hotel_names_path, encoding="utf-8") as f:
            all_hotels = json.load(f)
        print(f"  {len(all_hotels):,} hotels in hotel_names.json — counting sentences per hotel ...")
        hotel_counts: dict[str, int] = {}
        for i, hotel in enumerate(all_hotels):
            result = src_ev.get(
                where={"hotel_name": hotel},
                include=[],
            )
            hotel_counts[hotel] = len(result["ids"])
            if (i + 1) % 100 == 0:
                print(f"  Counted {i + 1:,} / {len(all_hotels):,} hotels ...", end="\r")
        print(f"  Counted {len(all_hotels):,} hotels.                        ")
    else:
        # Fall back to full metadata scan if hotel_names.json is missing
        print("  hotel_names.json not found — scanning all metadata (slow) ...")
        hotel_counts = {}
        batch_size = 5000
        offset = 0
        while offset < total_docs:
            batch = src_ev.get(limit=batch_size, offset=offset, include=["metadatas"])
            for meta in batch["metadatas"]:
                h = meta.get("hotel_name", "")
                hotel_counts[h] = hotel_counts.get(h, 0) + 1
            offset += batch_size
            print(f"  Scanned {min(offset, total_docs):,} / {total_docs:,}", end="\r")
        print(f"\n  Found {len(hotel_counts):,} hotels.")

    # ── Pick top N ───────────────────────────────────────────────────────────
    top_hotels = sorted(hotel_counts, key=hotel_counts.__getitem__, reverse=True)[:n_hotels]
    selected_count = sum(hotel_counts[h] for h in top_hotels)
    print(f"\nTop {n_hotels} hotels selected ({selected_count:,} sentences):")
    for i, h in enumerate(top_hotels[:5], 1):
        print(f"  {i}. {h}  ({hotel_counts[h]:,} sentences)")
    if n_hotels > 5:
        print(f"  ... and {n_hotels - 5} more")

    # ── Destination client ───────────────────────────────────────────────────
    if os.path.exists(DEMO_DB_DIR):
        print(f"\nRemoving existing {DEMO_DB_DIR} ...")
        shutil.rmtree(DEMO_DB_DIR)
    os.makedirs(DEMO_DB_DIR)
    dst_client = chromadb.PersistentClient(path=str(DEMO_DB_DIR))

    # ── Copy evidence_store ──────────────────────────────────────────────────
    print(f"\nCopying evidence_store for {n_hotels} hotels ...")
    UPSERT_BATCH = 5000

    dst_ev = dst_client.create_collection(
        name="evidence_store",
        metadata={"hnsw:space": "cosine"},
    )
    for i, hotel in enumerate(top_hotels):
        result = src_ev.get(
            where={"hotel_name": hotel},
            include=["documents", "metadatas", "embeddings"],
        )
        ids, docs, metas, embeds = (
            result["ids"], result["documents"],
            result["metadatas"], result["embeddings"],
        )
        for start in range(0, len(ids), UPSERT_BATCH):
            end = start + UPSERT_BATCH
            dst_ev.upsert(
                ids=ids[start:end],
                documents=docs[start:end],
                metadatas=metas[start:end],
                embeddings=embeds[start:end],
            )
        print(f"  [{i + 1:>2}/{n_hotels}] {hotel[:60]:60s}  {len(ids):>5} docs")
    print(f"  evidence_store done: {dst_ev.count():,} documents.")

    # ── Copy summary_store ───────────────────────────────────────────────────
    print(f"\nCopying summary_store ...")
    src_sum = src_client.get_collection("summary_store")
    dst_sum = dst_client.create_collection(
        name="summary_store",
        metadata={"hnsw:space": "cosine"},
    )

    copy_ids = ["sum___global__"] + [f"sum_{h}" for h in top_hotels]
    result = src_sum.get(
        ids=copy_ids,
        include=["documents", "metadatas", "embeddings"],
    )
    if result["ids"]:
        dst_sum.upsert(
            ids=result["ids"],
            documents=result["documents"],
            metadatas=result["metadatas"],
            embeddings=result["embeddings"],
        )
    print(f"  summary_store done: {dst_sum.count():,} documents.")

    # ── Write sentinel ───────────────────────────────────────────────────────
    with open(os.path.join(DEMO_DB_DIR, ".ingest_complete"), "w") as f:
        f.write("ok\n")

    # ── Write filtered hotel list ────────────────────────────────────────────
    demo_names_path = os.path.join(OUTPUT_DIR, "hotel_names_demo.json")
    with open(demo_names_path, "w", encoding="utf-8") as f:
        json.dump(sorted(top_hotels), f, indent=2)
    print(f"\n  Wrote {demo_names_path}  ({n_hotels} hotels)")

    print(f"\nDone.  Demo DB: {DEMO_DB_DIR}")
    print(
        "\nNext steps for HF Spaces deployment:\n"
        "  1. Copy chromadb_demo/ as chromadb/ into your Space repo\n"
        "  2. Copy outputs/hotel_names_demo.json as outputs/hotel_names.json in Space repo\n"
        "  3. Push to HF Spaces with git lfs"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build demo ChromaDB for HF Spaces")
    parser.add_argument(
        "--hotels", type=int, default=50,
        help="Number of top hotels by review count to include (default: 50)",
    )
    args = parser.parse_args()
    build_demo_db(args.hotels)
