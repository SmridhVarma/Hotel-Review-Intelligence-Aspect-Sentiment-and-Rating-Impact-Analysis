# fix_cleanliness_noise.py — Stage 2/3 one-off fix | Module A (ABSA): Re-curates the cleanliness and noise keyword lists to remove LDA noise and cross-aspect bleed.
#
# Input:  outputs/aspect_dictionary.json
# Output: outputs/aspect_dictionary.json (updated in-place)

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

JSON_PATH = Path("outputs/aspect_dictionary.json")
BACKUP_PATH = Path("outputs/aspect_dictionary.backup.json")

# Seed-first vocabularies for the two under-represented aspects. These come
# from domain knowledge of hotel reviews: distinctive surface forms that
# almost always indicate the aspect, ordered by signal strength.
CLEANLINESS_SEEDS = [
    "clean", "cleanliness", "dirty", "filthy", "spotless", "tidy",
    "stain", "stained", "stains",
    "dust", "dusty",
    "hygiene", "hygienic", "unhygienic",
    "smell", "smelly", "smells", "stink", "stinks", "stinky",
    "musty", "mould", "mold", "moldy", "mouldy",
    "hair", "hairs",
    "sheets", "towels", "linen", "linens",
    "fresh", "spotlessly clean", "immaculate",
]

NOISE_SEEDS = [
    "noise", "noisy", "quiet", "loud", "loudly",
    "soundproof", "soundproofing", "soundproofed",
    "thin walls", "street noise", "traffic noise", "road noise",
    "traffic", "music", "construction",
    "sleep", "sleeping", "asleep", "awake",
    "disturbed", "disturbing", "disturbance",
    "peaceful", "silent", "silence",
    "bar noise", "club noise", "party",
    "snore", "snoring",
]

# Words that LDA bled into Topic 0 (cleanliness slot) and Topic 4 (noise
# slot) but don't actually belong to those aspects. Filter these out when
# we add LDA topic words.
GENERIC_PRAISE_BLOCKLIST = {
    "location", "staff", "friendly", "helpful", "comfortable", "room",
    "rooms", "excellent", "breakfast", "perfect", "modern", "spacious",
    "bed", "bathroom", "shower", "size", "floor", "view", "little",
    "double", "big", "beds", "comfy", "space", "pillows", "bath", "window",
    "small", "room small", "staff friendly", "friendly staff",
    "friendly helpful", "staff helpful", "helpful staff",
}

# A few LDA-derived words ARE relevant to cleanliness/noise and should be
# kept if they appear in the topic. Allowlist them.
CLEANLINESS_LDA_ALLOWLIST = {"clean", "tidy", "fresh"}
NOISE_LDA_ALLOWLIST = {"noise", "quiet"}


def dedupe_preserve_order(items):
    """Keep first occurrence, lowercase, preserve order."""
    seen = set()
    out = []
    for x in items:
        k = x.lower().strip()
        if k and k not in seen:
            seen.add(k)
            out.append(k)
    return out


def rebuild_aspect(seeds, lda_topic_words, lda_allowlist):
    """Seeds first, then any LDA topic words that are on the allowlist."""
    kept_lda = [w for w in lda_topic_words
                if w.lower() in lda_allowlist
                and w.lower() not in GENERIC_PRAISE_BLOCKLIST]
    return dedupe_preserve_order(list(seeds) + kept_lda)


def main():
    if not JSON_PATH.exists():
        raise SystemExit(f"Not found: {JSON_PATH}. Run from the repo root.")

    # Backup before modifying.
    shutil.copy(JSON_PATH, BACKUP_PATH)
    print(f"Backed up original -> {BACKUP_PATH}")

    with JSON_PATH.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    aspects = payload["aspects"]
    raw_topics = payload.get("metadata", {}).get("raw_topic_top_words", {})
    mapping = payload.get("metadata", {}).get("topic_to_aspect", {})

    # Look up which raw topic was assigned to each of the two aspects.
    cleanliness_topic_idx = mapping.get("cleanliness")
    noise_topic_idx = mapping.get("noise")
    cleanliness_lda = raw_topics.get(str(cleanliness_topic_idx), []) \
        if cleanliness_topic_idx is not None else []
    noise_lda = raw_topics.get(str(noise_topic_idx), []) \
        if noise_topic_idx is not None else []

    old_cleanliness = aspects["cleanliness"]
    old_noise = aspects["noise"]

    aspects["cleanliness"] = rebuild_aspect(
        CLEANLINESS_SEEDS, cleanliness_lda, CLEANLINESS_LDA_ALLOWLIST
    )
    aspects["noise"] = rebuild_aspect(
        NOISE_SEEDS, noise_lda, NOISE_LDA_ALLOWLIST
    )

    # Provenance: record what changed and why.
    payload.setdefault("metadata", {})
    payload["metadata"]["post_processing"] = {
        "applied_at": datetime.now(timezone.utc).isoformat(),
        "rationale": (
            "LDA Topic 0 (mapped to cleanliness) and Topic 4 (mapped to "
            "noise) were dominated by generic-praise and room-amenity "
            "vocabulary, masking the actual cleanliness and noise signal. "
            "Rebuilt these two aspects with seed-first ordering plus an "
            "allowlist of LDA terms ('clean', 'tidy', 'fresh' for "
            "cleanliness; 'noise', 'quiet' for noise). Other four aspects "
            "left unchanged."
        ),
        "cleanliness": {
            "previous_size": len(old_cleanliness),
            "new_size": len(aspects["cleanliness"]),
        },
        "noise": {
            "previous_size": len(old_noise),
            "new_size": len(aspects["noise"]),
        },
    }

    with JSON_PATH.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"Updated -> {JSON_PATH}")
    print()
    print("New cleanliness keywords (first 15):")
    print("  " + ", ".join(aspects["cleanliness"][:15]))
    print()
    print("New noise keywords (first 15):")
    print("  " + ", ".join(aspects["noise"][:15]))


if __name__ == "__main__":
    main()
