"""
One-off fix: remove noisy LDA words from the `staff` and `food` aspects
in outputs/aspect_dictionary.json.

Why this script exists
----------------------
The fix_cleanliness_noise.py script already corrected cleanliness and noise.
Staff and food were left unchanged, but their LDA topics were also noisy:

  Staff (LDA Topic 2): top words included "room", "rooms", "check", "didn",
    "like", "wifi", "booking", "asked", "wasn", "morning", "pool", "booked",
    "people", "pay", "early", "door", "don" — generic interaction/booking
    vocabulary, not staff-quality signal. Critically, "room" and "rooms" are
    claimed by staff before the room aspect (first-seen-wins in
    build_keyword_index), so sentences like "the room was spacious" get
    mis-classified as staff.

  Food (LDA Topic 3): top words included "staff", "service", "lovely",
    "amazing", "excellent", "loved", "best", "view", "fantastic", "free",
    "welcoming", "beautiful", "wonderful", "area", "reception" — generic
    praise vocabulary. These words either already belong to other aspects
    or add noise without food signal.

What this script does
---------------------
Strips the noisy LDA words from staff and food, keeping only the
seed-derived keywords that have clear aspect signal. Other four aspects
are left unchanged.

Run from the repo root:
    python scripts/fix_staff_food.py
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

JSON_PATH = Path("outputs/aspect_dictionary.json")
BACKUP_PATH = Path("outputs/aspect_dictionary.backup2.json")

# LDA noise words to strip from staff — generic booking/interaction tokens
# that appeared in Topic 2 but carry no staff-quality signal.
# "room"/"rooms" are the critical ones: they were stealing room-aspect
# sentences via first-seen-wins keyword ownership.
STAFF_NOISE = {
    "room", "rooms", "check", "didn", "like", "booking", "wifi",
    "asked", "wasn", "morning", "pool", "booked", "people", "pay",
    "early", "door", "don",
}

# LDA noise words to strip from food — generic praise tokens from Topic 3
# that have no food-specific meaning.
FOOD_NOISE = {
    "staff", "service", "lovely", "room", "amazing", "excellent",
    "loved", "best", "view", "fantastic", "free", "welcoming",
    "beautiful", "wonderful", "area", "reception",
}


def main() -> None:
    if not JSON_PATH.exists():
        raise SystemExit(f"Not found: {JSON_PATH}. Run from the repo root.")

    shutil.copy(JSON_PATH, BACKUP_PATH)
    print(f"Backed up original -> {BACKUP_PATH}")

    with JSON_PATH.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    aspects = payload["aspects"]

    old_staff = aspects["staff"][:]
    old_food = aspects["food"][:]

    aspects["staff"] = [w for w in old_staff if w.lower().strip() not in STAFF_NOISE]
    aspects["food"]  = [w for w in old_food  if w.lower().strip() not in FOOD_NOISE]

    removed_staff = [w for w in old_staff if w.lower().strip() in STAFF_NOISE]
    removed_food  = [w for w in old_food  if w.lower().strip() in FOOD_NOISE]

    payload.setdefault("metadata", {})
    payload["metadata"].setdefault("post_processing_2", {})
    payload["metadata"]["post_processing_2"] = {
        "applied_at": datetime.now(timezone.utc).isoformat(),
        "rationale": (
            "Staff (LDA Topic 2) contained generic booking/interaction noise "
            "words including 'room'/'rooms', which caused keyword ownership "
            "conflicts — staff claimed 'room' before the room aspect, "
            "mis-classifying room-only sentences as staff. "
            "Food (LDA Topic 3) contained generic praise vocabulary with no "
            "food-specific signal. Both aspects stripped to seed-derived "
            "keywords only."
        ),
        "staff": {
            "previous_size": len(old_staff),
            "new_size": len(aspects["staff"]),
            "removed": removed_staff,
        },
        "food": {
            "previous_size": len(old_food),
            "new_size": len(aspects["food"]),
            "removed": removed_food,
        },
    }

    with JSON_PATH.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"Updated -> {JSON_PATH}")
    print(f"\nStaff: {len(old_staff)} -> {len(aspects['staff'])} keywords")
    print(f"  Removed: {removed_staff}")
    print(f"\nFood: {len(old_food)} -> {len(aspects['food'])} keywords")
    print(f"  Removed: {removed_food}")
    print(f"\nStaff keywords now: {aspects['staff']}")
    print(f"\nFood keywords now:  {aspects['food']}")


if __name__ == "__main__":
    main()
