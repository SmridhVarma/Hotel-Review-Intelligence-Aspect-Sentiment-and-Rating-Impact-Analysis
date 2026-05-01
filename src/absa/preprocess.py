# preprocess.py — Stage 1 | Module A (ABSA): Cleans raw review text, extracts reviewer segments, and sentence-tokenises each review.
#
# Input:  data/data.xlsx
# Output: outputs/sentences.csv, outputs/clean_reviews_stage1.csv

import os
import re
import ast

import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MIN_SENTENCE_CHARS = 3

# Whole-field placeholder patterns: matched case-insensitively after strip.
# These indicate the reviewer left the field blank in spirit (no useful text).
PLACEHOLDER_PATTERN = re.compile(
    r"^\s*("
    r"no positive|no negative|nothing|nothing really|nothing at all|"
    r"n/a|n a|na|none|nil|"
    r"no complaints?|"
    r"nothing to dislike|nothing to complain about"
    r")\s*$",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Text cleaning helpers  (importable by Stage 2 / 3 if needed)
# ---------------------------------------------------------------------------

def is_empty_review(text) -> bool:
    """Return True for NaN, blank strings, or known non-informative placeholders."""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return True
    t = str(text).strip()
    if not t:
        return True
    return bool(PLACEHOLDER_PATTERN.match(t))


def clean_review_text(text) -> str:
    """Strip, collapse internal whitespace, and return '' for placeholder values."""
    if pd.isna(text):
        return ""
    s = str(text).strip()
    s = re.sub(r"\s+", " ", s)
    if is_empty_review(s):
        return ""
    return s


def word_count_non_empty(text) -> int:
    """Word count of cleaned text; 0 for empty / placeholder strings."""
    if not text or is_empty_review(text):
        return 0
    return len(str(text).split())


# ---------------------------------------------------------------------------
# Reviewer segment extraction
# ---------------------------------------------------------------------------

def extract_reviewer_segment(tags) -> str:
    """Map the raw Tags field to one of: Business | Couple | Family | Solo | Group | Unknown."""
    if pd.isna(tags) or tags is None or str(tags).strip() == "":
        return "Unknown"
    raw = str(tags)
    try:
        items = ast.literal_eval(raw)
        blob = " ".join(str(x) for x in items).lower()
    except (ValueError, SyntaxError, TypeError):
        blob = raw.lower()

    # Order matters: more specific checks before 'couple'
    if "business trip" in blob:
        return "Business"
    if "family" in blob:
        return "Family"
    if "solo traveler" in blob:
        return "Solo"
    if "travelers with friends" in blob or re.search(r"\bgroup\b", blob):
        return "Group"
    if "couple" in blob:
        return "Couple"
    return "Unknown"


# ---------------------------------------------------------------------------
# Review-level helpers
# ---------------------------------------------------------------------------

def build_full_review_clean(pos: str, neg: str) -> str:
    """Concatenate non-empty cleaned positive and negative text into one field."""
    p = "" if is_empty_review(pos) else str(pos).strip()
    n = "" if is_empty_review(neg) else str(neg).strip()
    if p and n:
        return f"{p} {n}"
    return p or n


# ---------------------------------------------------------------------------
# Sentence-level table builder
# ---------------------------------------------------------------------------

def build_sentence_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Explode the review-level DataFrame into one row per sentence.

    Uses itertuples (faster than iterrows — avoids Series construction per row)
    over the cleaned-text columns.  Returns a DataFrame ready to write as
    outputs/sentences.csv.
    """
    _setup_nltk()

    rows = []
    field_specs = [
        ("positive_review_clean", "Positive_Review", "positive"),
        ("negative_review_clean", "Negative_Review", "negative"),
    ]

    for row in df.itertuples(index=False):
        for text_col, field_name, polarity in field_specs:
            text = getattr(row, text_col, "")
            if is_empty_review(text):
                continue
            t = str(text).strip()
            if not t:
                continue
            try:
                sents = sent_tokenize(t, language="english")
            except Exception:
                sents = re.split(r"(?<=[.!?])\s+", t)
            for sent in sents:
                s = sent.strip()
                if len(s) < MIN_SENTENCE_CHARS:
                    continue
                if PLACEHOLDER_PATTERN.match(s):
                    continue
                rows.append(
                    {
                        "review_id": row.review_id,
                        "hotel_name": row.Hotel_Name,
                        "reviewer_score": row.Reviewer_Score,
                        "reviewer_segment": row.reviewer_segment,
                        "review_date": row.Review_Date,
                        "sentence": s,
                        "source_field": field_name,
                        "source_polarity": polarity,
                    }
                )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# NLTK setup
# ---------------------------------------------------------------------------

def _setup_nltk() -> None:
    """Download punkt tokenizer data if not already present (runs once)."""
    for resource, pkg in [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab/english", "punkt_tab"),
    ]:
        try:
            nltk.data.find(resource)
        except LookupError:
            try:
                nltk.download(pkg, quiet=True)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(data_path: str = None, output_dir: str = None) -> None:
    """
    Execute the full Stage 1 pipeline.

    Parameters
    ----------
    data_path : str, optional
        Path to data.csv. Defaults to data/data.csv relative to the project root.
    output_dir : str, optional
        Directory for output files. Defaults to outputs/ relative to project root.
    """
    # Resolve paths relative to this file so the script works from any cwd
    here = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.normpath(os.path.join(here, "..", ".."))

    if data_path is None:
        # Accept either .xlsx or .csv — prefer .xlsx (OneDrive default)
        for _ext in ("data.xlsx", "data.csv"):
            _candidate = os.path.join(project_root, "data", _ext)
            if os.path.isfile(_candidate):
                data_path = _candidate
                break
        else:
            data_path = os.path.join(project_root, "data", "data.xlsx")
    if output_dir is None:
        output_dir = os.path.join(project_root, "outputs")

    os.makedirs(output_dir, exist_ok=True)

    # ── 1. Load ──────────────────────────────────────────────────────────────
    if not os.path.isfile(data_path):
        raise FileNotFoundError(
            f"Dataset not found at {data_path}\n"
            "Place data.xlsx (or data.csv) in the data/ folder."
        )

    print(f"Loading {data_path} …")
    if data_path.endswith(".xlsx"):
        df = pd.read_excel(data_path)
    else:
        df = pd.read_csv(data_path, low_memory=False)
    df["Review_Date"] = pd.to_datetime(df["Review_Date"], errors="coerce")
    df["Review_Date"] = df["Review_Date"].dt.strftime("%Y-%m-%d")
    print(f"  Loaded {len(df):,} rows, {df.shape[1]} columns.")

    # ── 2. Add review_id ─────────────────────────────────────────────────────
    if "review_id" not in df.columns:
        df["review_id"] = df.index

    # ── 3. Clean text fields ─────────────────────────────────────────────────
    print("Cleaning review text …")
    for src_col, out_col in [
        ("Positive_Review", "positive_review_clean"),
        ("Negative_Review", "negative_review_clean"),
    ]:
        if src_col in df.columns:
            df[out_col] = df[src_col].map(clean_review_text)
        else:
            print(f"  Warning: column {src_col!r} not found; filling {out_col!r} with empty strings.")
            df[out_col] = ""

    df["positive_word_count"] = df["positive_review_clean"].map(word_count_non_empty)
    df["negative_word_count"] = df["negative_review_clean"].map(word_count_non_empty)
    df["total_word_count"] = df["positive_word_count"] + df["negative_word_count"]

    # ── 4. Reviewer segment ───────────────────────────────────────────────────
    print("Extracting reviewer segments …")
    if "Tags" in df.columns:
        df["reviewer_segment"] = df["Tags"].map(extract_reviewer_segment)
    else:
        df["reviewer_segment"] = "Unknown"

    # ── 5. Full review text ───────────────────────────────────────────────────
    df["full_review_clean"] = [
        build_full_review_clean(p, n)
        for p, n in zip(df["positive_review_clean"], df["negative_review_clean"])
    ]

    # ── 6. Save review-level file ─────────────────────────────────────────────
    review_cols = [
        "review_id", "Hotel_Name", "Hotel_Address", "Reviewer_Nationality",
        "Review_Date", "Reviewer_Score", "Tags", "reviewer_segment",
        "positive_review_clean", "negative_review_clean",
        "positive_word_count", "negative_word_count", "total_word_count",
        "full_review_clean",
    ]
    available = [c for c in review_cols if c in df.columns]
    review_level_path = os.path.join(output_dir, "clean_reviews_stage1.csv")
    df[available].to_csv(review_level_path, index=False, encoding="utf-8")
    print(f"  Saved review-level file: {review_level_path}  ({len(df):,} rows)")

    # ── 7. Build sentence-level table ─────────────────────────────────────────
    print("Sentence-tokenizing reviews (this takes a few minutes) …")
    sentence_df = build_sentence_rows(df)

    # ── 8. Save sentence-level file (primary pipeline input) ─────────────────
    sentences_path = os.path.join(output_dir, "sentences.csv")
    sentence_df.to_csv(sentences_path, index=False, encoding="utf-8")
    print(f"  Saved sentences file:     {sentences_path}  ({len(sentence_df):,} rows)")

    # ── 9. Validation ─────────────────────────────────────────────────────────
    _validate(df, sentence_df)
    print("\nStage 1 complete.")


def _validate(review_df: pd.DataFrame, sentence_df: pd.DataFrame) -> None:
    """Lightweight assertions — fail fast if something is wrong before downstream stages run."""
    # No placeholder residue in cleaned text columns
    for col in ("positive_review_clean", "negative_review_clean", "full_review_clean"):
        residue = int(
            review_df[col]
            .fillna("")
            .astype(str)
            .map(lambda t: bool(t) and bool(PLACEHOLDER_PATTERN.match(t)))
            .sum()
        )
        assert residue == 0, f"Unexpected placeholder residue in {col}: {residue} rows"

    # Sentence file must have required columns
    required = {
        "review_id", "hotel_name", "reviewer_score", "reviewer_segment",
        "review_date", "sentence", "source_field", "source_polarity",
    }
    missing_cols = required - set(sentence_df.columns)
    assert not missing_cols, f"sentences.csv missing columns: {missing_cols}"

    # No blank sentences
    assert sentence_df["sentence"].notna().all(), "Null sentences found"
    assert (sentence_df["sentence"].astype(str).str.len() > 0).all(), "Empty sentences found"

    # Both polarities present (sanity check on the split)
    assert (sentence_df["source_polarity"] == "positive").any(), "No positive-polarity sentences"
    assert (sentence_df["source_polarity"] == "negative").any(), "No negative-polarity sentences"

    print("  Validation passed.")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run()
