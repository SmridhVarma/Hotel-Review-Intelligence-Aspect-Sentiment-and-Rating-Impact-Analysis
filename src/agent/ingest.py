# =============================================================================
# ingest.py — ChromaDB Ingestion (Stage 5)
# =============================================================================
# Purpose:
#   Embeds all review sentences and SHAP summaries into a local ChromaDB
#   instance for retrieval by the LangGraph agent. Run once after the full
#   pipeline completes. Safe to re-run — clears and rebuilds both collections.
#
#   evidence_store collection:
#     One document per sentence. Embedding model: text-embedding-3-small.
#     Metadata: hotel_name (str), aspect (str), sentiment (int),
#               reviewer_segment (str), reviewer_score (float)
#     Used by: evidence_retriever, segment_filter nodes
#
#   summary_store collection:
#     One document per hotel + one "__global__" document.
#     Document text: human-readable SHAP narrative, e.g.
#       "For Hotel Arena: cleanliness has the largest negative impact (-0.42),
#        staff has the largest positive impact (+0.31)..."
#     Metadata: hotel_name (str)
#     Used by: summary_retriever node
#
# Input:
#   outputs/aspect_sentences.csv  (from sentiment_assignment.py)
#     review_id, hotel_name, sentence, aspect, sentiment,
#     reviewer_segment, reviewer_score
#
#   outputs/shap_summary.json  (from model.py)
#     list of { hotel_name, aspect_impacts: { aspect: shap_value } }
#
#   .env  →  OPENAI_API_KEY
#
# Output:
#   chromadb/  — local persistent ChromaDB directory (gitignored)
#     collections: evidence_store, summary_store
#
# Cost estimate:
#   ~1M sentences × ~20 tokens avg = ~20M tokens
#   text-embedding-3-small at $0.02/1M tokens ≈ $0.40 total
#   Runtime: ~30–60 min depending on OpenAI rate limits
#
# Tech stack (locked):
#   chromadb, openai, python-dotenv, pandas
# =============================================================================
