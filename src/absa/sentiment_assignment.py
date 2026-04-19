# sentiment_assignment.py — Stage 3: Aspect-Level Sentiment Assignment
#
# Purpose:
#   Keyword-matches each sentence to an aspect, then assigns sentiment
#   from source polarity. Produces both a sentence-level labeled dataset
#   and a review-level feature matrix for downstream modeling.
#
# Input:
#   outputs/sentences.csv
#     review_id        (int)  : review identifier
#     hotel_name       (str)  : hotel identifier
#     sentence         (str)  : sentence text
#     source_polarity  (str)  : "positive" | "negative"
#     reviewer_segment (str)  : guest segment
#     reviewer_score   (float): numeric rating
#
#   outputs/aspect_dictionary.json
#     Schema: { aspect_name: [keyword, ...] }
#
# Output:
#   outputs/aspect_sentences.csv
#     review_id        (int)  : review identifier
#     hotel_name       (str)  : hotel identifier
#     sentence         (str)  : sentence text
#     aspect           (str)  : matched aspect | null if no match
#     sentiment        (int)  : +1 (positive) | -1 (negative) | 0 (no match)
#     reviewer_segment (str)  : guest segment
#     reviewer_score   (float): numeric rating
#
#   outputs/review_features.csv
#     review_id    (int)  : review identifier
#     hotel_name   (str)  : hotel identifier
#     reviewer_score (float): numeric rating — prediction target for model.py
#     cleanliness  (int)  : +1 | -1 | 0
#     staff        (int)  : +1 | -1 | 0
#     location     (int)  : +1 | -1 | 0
#     noise        (int)  : +1 | -1 | 0
#     food         (int)  : +1 | -1 | 0
#     room         (int)  : +1 | -1 | 0
#
# ─── DOWNSTREAM REQUIREMENTS (for src/agent/ingest.py) ──────────────────────
#
# The agent's ChromaDB ingest reads outputs/aspect_sentences.csv directly.
# For metadata filters in the retriever node to work correctly, the CSV must
# satisfy the following format contract EXACTLY — mismatches cause silent
# zero-result retrieval:
#
#   hotel_name (str):
#     Must be the EXACT string from the original data.csv Hotel_Name column.
#     Do not normalise casing or strip whitespace — the agent fuzzy-matches
#     user queries against this exact list, so any transformation here breaks
#     hotel_resolver resolution downstream.
#
#   aspect (str):
#     Must be Title-Cased: "Cleanliness" | "Staff" | "Location" |
#                          "Noise" | "Food" | "Room"
#     The retriever filters on aspect using ChromaDB's $in operator. Lowercase
#     or inconsistent casing means the filter returns no results.
#
#   sentiment (str):
#     Must be converted from the integer representation (+1/-1/0) to:
#       +1 -> "Positive"
#        0 -> "Neutral"
#       -1 -> "Negative"
#     The retriever uses sentiment for stratified retrieval
#     (e.g. top-7 Positive + top-7 Negative + top-7 Neutral per neutral query).
#     Integer values will not match the filter and collapse retrieval to zero.
#
#   reviewer_segment (str):
#     Must be one of exactly: "Business" | "Couple" | "Family" | "Solo" | "Group"
#     These are the segment filter keys used in ChromaDB metadata queries.
#
#   Rows where aspect is null or sentiment is 0 (no match):
#     These should be EXCLUDED from outputs/aspect_sentences.csv or clearly
#     flagged — ingest.py will skip them, but including them inflates the
#     collection with un-filterable noise documents.
