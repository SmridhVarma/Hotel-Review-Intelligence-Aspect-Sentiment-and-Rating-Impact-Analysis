# paths.py — Canonical pipeline file paths
#
# Single source of truth for all input/output paths.
# Import this module from any stage file:
#
#   import sys, os
#   _SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#   if _SRC not in sys.path: sys.path.insert(0, _SRC)
#   from paths import SENTENCES, ASPECT_DICTIONARY   # etc.

import os

# Project root: parent of src/
PROJECT_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
)

DATA_DIR      = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR    = os.path.join(PROJECT_ROOT, "outputs")
ARTIFACTS_DIR = os.path.join(OUTPUT_DIR, "model_artifacts")

# ── Stage 1 inputs / outputs  (preprocess.py) ────────────────────────────────
RAW_DATA      = os.path.join(DATA_DIR,    "data.xlsx")
SENTENCES     = os.path.join(OUTPUT_DIR,  "sentences.csv")
CLEAN_REVIEWS = os.path.join(OUTPUT_DIR,  "clean_reviews_stage1.csv")

# ── Stage 2 output  (aspect_extraction.py) ───────────────────────────────────
ASPECT_DICTIONARY = os.path.join(OUTPUT_DIR, "aspect_dictionary.json")

# ── Stage 3 outputs  (sentiment_assignment.py) ───────────────────────────────
ASPECT_SENTENCES = os.path.join(OUTPUT_DIR, "aspect_sentences.csv")
REVIEW_FEATURES  = os.path.join(OUTPUT_DIR, "review_features.csv")

# ── Stage 4 outputs  (model.py) ──────────────────────────────────────────────
LINEAR_MODEL  = os.path.join(ARTIFACTS_DIR, "linear_model.pkl")
XGB_MODEL     = os.path.join(ARTIFACTS_DIR, "xgb_model.pkl")
SHAP_SUMMARY  = os.path.join(OUTPUT_DIR,    "shap_summary.json")
IMPACT_REPORT = os.path.join(OUTPUT_DIR,    "impact_report.csv")

# ── Stage 4 outputs  (evaluate.py) ───────────────────────────────────────────
EVALUATION_REPORT = os.path.join(OUTPUT_DIR, "evaluation_report.json")

# ── Stage 5 inputs  (ingest.py) ──────────────────────────────────────────────
# ASPECT_SENTENCES and SHAP_SUMMARY above — reused by agent ingest
CHROMADB_DIR      = os.path.join(PROJECT_ROOT, "chromadb")
DEMO_VECTORS_DIR  = os.path.join(PROJECT_ROOT, "demo_vectors")  # numpy fallback for HF Spaces
