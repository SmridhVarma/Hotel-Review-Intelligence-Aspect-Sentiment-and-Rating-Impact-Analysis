# sentiment_assignment.py — Stage 3 | Module A (ABSA): Keyword-matches sentences to aspects and assigns sentiment; aggregates to a review-level feature matrix.
#
# Input:  outputs/sentences.csv, outputs/aspect_dictionary.json
# Output: outputs/aspect_sentences.csv, outputs/review_features.csv

import os
import sys
import json
import re

import pandas as pd
import numpy as np
from tqdm import tqdm

# ── Path setup ────────────────────────────────────────────────────────────────
_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from paths import (
    SENTENCES,
    ASPECT_DICTIONARY,
    ASPECT_SENTENCES,
    REVIEW_FEATURES,
)

# ── Constants ─────────────────────────────────────────────────────────────────

ASPECTS = ["cleanliness", "staff", "location", "noise", "food", "room"]

POLARITY_TO_SENTIMENT = {
    "positive": "Positive",
    "negative": "Negative",
}

SENTIMENT_TO_INT = {
    "Positive": 1,
    "Negative": -1,
    "Neutral": 0,
}


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_sentences() -> pd.DataFrame:
    """Load the Stage 1 sentence corpus."""
    if not os.path.isfile(SENTENCES):
        raise FileNotFoundError(
            f"sentences.csv not found at {SENTENCES}\n"
            "Run Stage 1 (preprocess.py) first."
        )
    return pd.read_csv(SENTENCES, low_memory=False)


def load_aspect_dictionary() -> dict:
    """Load the Stage 2 aspect keyword dictionary and return the aspects sub-dict."""
    if not os.path.isfile(ASPECT_DICTIONARY):
        raise FileNotFoundError(
            f"aspect_dictionary.json not found at {ASPECT_DICTIONARY}\n"
            "Run Stage 2 (aspect_extraction.py) first."
        )
    with open(ASPECT_DICTIONARY, encoding="utf-8") as f:
        data = json.load(f)
    # Support both raw list and nested {"aspects": {...}} schema
    if "aspects" in data:
        return data["aspects"]
    return data


# ── Keyword index builder ─────────────────────────────────────────────────────

def build_keyword_index(aspect_dict: dict) -> dict:
    """
    Build a sorted keyword → aspect mapping for efficient matching.

    Multi-word phrases (e.g. "air conditioning") are matched before
    single-word keywords to prevent partial overlaps.

    Returns
    -------
    dict with two keys:
        "phrases"  : list of (phrase, aspect) tuples, longest-first
        "words"    : dict { word: aspect } for single-token lookup
    """
    phrases = []   # (phrase_text, aspect)
    words = {}     # word -> aspect  (last write wins for overlaps)

    for aspect, keywords in aspect_dict.items():
        aspect_lower = aspect.lower()
        for kw in keywords:
            kw_clean = kw.strip().lower()
            if not kw_clean:
                continue
            if " " in kw_clean:
                phrases.append((kw_clean, aspect_lower))
            else:
                # Only overwrite if not yet assigned (preserves aspect priority order)
                words.setdefault(kw_clean, aspect_lower)

    # Sort phrases longest-first so "air conditioning" beats "air"
    phrases.sort(key=lambda x: len(x[0]), reverse=True)

    return {"phrases": phrases, "words": words}


# ── Aspect detection ──────────────────────────────────────────────────────────

def detect_aspect(sentence: str, keyword_index: dict) -> str | None:
    """
    Return the best-matching aspect for a sentence, or None if no match.

    Strategy:
      1. Lowercase the sentence.
      2. Check multi-word phrases first (longest-first order).
      3. Tokenise and check single-word keywords.
      4. If multiple aspects match, return the one with the most keyword hits.
         Ties are broken by ASPECTS list order (earlier = higher priority).
    """
    text = sentence.lower()
    hit_counts: dict[str, int] = {}

    # 1. Multi-word phrase scan
    for phrase, aspect in keyword_index["phrases"]:
        if phrase in text:
            hit_counts[aspect] = hit_counts.get(aspect, 0) + 1

    # 2. Single-word token scan
    tokens = set(re.findall(r"[a-z]+", text))
    word_map = keyword_index["words"]
    for token in tokens:
        if token in word_map:
            asp = word_map[token]
            hit_counts[asp] = hit_counts.get(asp, 0) + 1

    if not hit_counts:
        return None

    # Pick aspect with highest hit count; break ties by ASPECTS list order
    best = max(
        hit_counts.keys(),
        key=lambda a: (hit_counts[a], -ASPECTS.index(a) if a in ASPECTS else -99),
    )
    return best


# ── Core assignment ───────────────────────────────────────────────────────────

def assign_sentiment(sentences_df: pd.DataFrame, aspect_dict: dict) -> pd.DataFrame:
    """
    Keyword-match each sentence to an aspect and assign sentiment.

    Parameters
    ----------
    sentences_df : DataFrame from sentences.csv
    aspect_dict  : { aspect_name: [keyword, ...] }

    Returns
    -------
    DataFrame with columns matching outputs/aspect_sentences.csv schema.
    Rows with no aspect match are EXCLUDED (per agent ingest contract).
    """
    keyword_index = build_keyword_index(aspect_dict)

    aspects_out   = []
    sentiments_out = []

    print("Assigning aspects and sentiments …")
    for row in tqdm(sentences_df.itertuples(index=False), total=len(sentences_df)):
        sentence = str(row.sentence)
        polarity = str(row.source_polarity).strip().lower()

        aspect = detect_aspect(sentence, keyword_index)
        if aspect is None:
            aspects_out.append(None)
            sentiments_out.append("Neutral")
        else:
            aspects_out.append(aspect.capitalize())          # Title-case for agent
            sentiments_out.append(POLARITY_TO_SENTIMENT.get(polarity, "Neutral"))

    sentences_df = sentences_df.copy()
    sentences_df["aspect"]    = aspects_out
    sentences_df["sentiment"] = sentiments_out

    # ── Build aspect_sentences.csv (exclude unmatched rows) ──────────────────
    matched = sentences_df[sentences_df["aspect"].notna()].copy()

    aspect_sentences = matched[[
        "review_id",
        "hotel_name",
        "sentence",
        "aspect",
        "sentiment",
        "reviewer_segment",
        "reviewer_score",
    ]].reset_index(drop=True)

    return aspect_sentences


# ── Review-level feature aggregation ─────────────────────────────────────────

def build_review_features(aspect_sentences: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate sentence-level aspect sentiments to one row per review.

    For each (review_id, aspect), compute the mean of integer sentiment
    values (+1 / -1). Reviews where an aspect is never mentioned get 0.

    Returns
    -------
    DataFrame with columns:
        review_id, hotel_name, reviewer_score,
        cleanliness, staff, location, noise, food, room
    """
    # Map sentiment string → int for arithmetic
    aspect_sentences = aspect_sentences.copy()
    aspect_sentences["sentiment_int"] = aspect_sentences["sentiment"].map(SENTIMENT_TO_INT)

    # Pivot: mean sentiment per (review_id, aspect)
    pivot = (
        aspect_sentences
        .groupby(["review_id", "aspect"])["sentiment_int"]
        .mean()
        .unstack(fill_value=0)
    )

    # Ensure all 6 aspect columns exist even if absent in data
    for asp in ASPECTS:
        col = asp.capitalize()
        if col not in pivot.columns:
            pivot[col] = 0

    # Rename columns to lowercase for model.py
    pivot.columns = [c.lower() for c in pivot.columns]
    pivot = pivot.reset_index()

    # Merge hotel_name and reviewer_score back in (from the first sentence per review)
    meta = (
        aspect_sentences[["review_id", "hotel_name", "reviewer_score"]]
        .drop_duplicates(subset="review_id")
    )
    review_features = meta.merge(pivot, on="review_id", how="left")

    # Guarantee column order
    feature_cols = ["review_id", "hotel_name", "reviewer_score"] + ASPECTS
    review_features = review_features[feature_cols].reset_index(drop=True)

    return review_features


# ── Validation ────────────────────────────────────────────────────────────────

def _validate(aspect_sentences: pd.DataFrame, review_features: pd.DataFrame) -> None:
    """Lightweight contract checks — fail fast before downstream stages run."""

    # aspect_sentences schema
    required_sent = {"review_id", "hotel_name", "sentence", "aspect", "sentiment",
                     "reviewer_segment", "reviewer_score"}
    missing = required_sent - set(aspect_sentences.columns)
    assert not missing, f"aspect_sentences.csv missing columns: {missing}"

    # Aspect values must be Title-Cased known aspects
    valid_aspects = {a.capitalize() for a in ASPECTS}
    bad_aspects = set(aspect_sentences["aspect"].dropna().unique()) - valid_aspects
    assert not bad_aspects, f"Unexpected aspect values: {bad_aspects}"

    # Sentiment values must be exactly these three strings
    valid_sentiments = {"Positive", "Negative", "Neutral"}
    bad_sentiments = set(aspect_sentences["sentiment"].unique()) - valid_sentiments
    assert not bad_sentiments, f"Unexpected sentiment values: {bad_sentiments}"

    # No rows with null aspect in aspect_sentences (excluded upstream)
    assert aspect_sentences["aspect"].notna().all(), "Null aspects found in aspect_sentences"

    # review_features schema
    required_feat = {"review_id", "hotel_name", "reviewer_score"} | set(ASPECTS)
    missing_feat = required_feat - set(review_features.columns)
    assert not missing_feat, f"review_features.csv missing columns: {missing_feat}"

    print("  Validation passed.")


# ── Main ──────────────────────────────────────────────────────────────────────

def run(sentences_path: str = None,
        aspect_dict_path: str = None,
        output_dir: str = None) -> None:
    """
    Execute Stage 3.

    Parameters
    ----------
    sentences_path    : path to sentences.csv  (default: paths.SENTENCES)
    aspect_dict_path  : path to aspect_dictionary.json (default: paths.ASPECT_DICTIONARY)
    output_dir        : directory for output files (default: paths.OUTPUT_DIR)
    """
    # Allow path overrides for testing
    global SENTENCES, ASPECT_DICTIONARY, ASPECT_SENTENCES, REVIEW_FEATURES
    if sentences_path:
        SENTENCES = sentences_path
    if aspect_dict_path:
        ASPECT_DICTIONARY = aspect_dict_path
    if output_dir:
        ASPECT_SENTENCES = os.path.join(output_dir, "aspect_sentences.csv")
        REVIEW_FEATURES  = os.path.join(output_dir, "review_features.csv")
        os.makedirs(output_dir, exist_ok=True)

    # ── 1. Load inputs ────────────────────────────────────────────────────────
    print(f"Loading sentences from {SENTENCES} …")
    sentences_df = load_sentences()
    print(f"  Loaded {len(sentences_df):,} sentences.")

    print(f"Loading aspect dictionary from {ASPECT_DICTIONARY} …")
    aspect_dict = load_aspect_dictionary()
    print(f"  Loaded {len(aspect_dict)} aspects: {list(aspect_dict.keys())}")

    # ── 2. Assign aspects + sentiments ────────────────────────────────────────
    aspect_sentences = assign_sentiment(sentences_df, aspect_dict)

    match_rate = len(aspect_sentences) / len(sentences_df) * 100
    print(f"  Matched {len(aspect_sentences):,} / {len(sentences_df):,} sentences "
          f"({match_rate:.1f}% match rate).")

    # Distribution summary
    print("\n  Aspect distribution:")
    print(aspect_sentences["aspect"].value_counts().to_string())
    print("\n  Sentiment distribution:")
    print(aspect_sentences["sentiment"].value_counts().to_string())

    # ── 3. Build review-level feature matrix ──────────────────────────────────
    print("\nBuilding review-level feature matrix …")
    review_features = build_review_features(aspect_sentences)
    print(f"  Built {len(review_features):,} review feature rows.")

    # ── 4. Validate ───────────────────────────────────────────────────────────
    print("\nValidating outputs …")
    _validate(aspect_sentences, review_features)

    # ── 5. Save ───────────────────────────────────────────────────────────────
    aspect_sentences.to_csv(ASPECT_SENTENCES, index=False, encoding="utf-8")
    print(f"\n  Saved: {ASPECT_SENTENCES}  ({len(aspect_sentences):,} rows)")

    review_features.to_csv(REVIEW_FEATURES, index=False, encoding="utf-8")
    print(f"  Saved: {REVIEW_FEATURES}  ({len(review_features):,} rows)")

    print("\nStage 3 complete.")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run()