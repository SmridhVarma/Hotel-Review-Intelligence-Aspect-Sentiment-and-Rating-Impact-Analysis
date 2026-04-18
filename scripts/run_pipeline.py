# =============================================================================
# run_pipeline.py — Full Pipeline Orchestrator
# =============================================================================
# Purpose:
#   Runs all pipeline stages end-to-end. Each stage is a separate import
#   and function call — stages can be skipped or re-run individually.
#
#   Stage 1  src.absa.preprocess          data/data.csv
#                                       → outputs/sentences.csv
#   Stage 2  src.absa.aspect_extraction   outputs/sentences.csv
#                                       → outputs/aspect_dictionary.json
#   Stage 3  src.absa.sentiment_assignment outputs/sentences.csv +
#                                          outputs/aspect_dictionary.json
#                                       → outputs/aspect_sentences.csv
#                                       → outputs/review_features.csv
#   Stage 4  src.rating_impact.model      outputs/review_features.csv
#                                       → outputs/model_artifacts/
#                                       → outputs/shap_summary.json
#   Stage 5  src.agent.ingest             outputs/aspect_sentences.csv +
#                                         outputs/shap_summary.json
#                                       → chromadb/
#
# Usage:
#   python scripts/run_pipeline.py              # full pipeline
#   python scripts/run_pipeline.py --from 3     # resume from stage 3
#   python scripts/run_pipeline.py --only 4     # run only stage 4
#
# Input:  data/data.csv
# Output: all outputs/ artifacts + chromadb/
# =============================================================================
