# =============================================================================
# evidence_retriever.py — Node: Evidence Store Retrieval
# =============================================================================
# Purpose:
#   Queries the ChromaDB evidence_store using the HyDE embedding from
#   hyde_expander. Applies optional metadata filters for hotel_name and
#   reviewer segment. Returns top-k sentence chunks with full metadata.
#
# Reads from state:
#   hyde_embedding  (list[float])  query vector from hyde_expander
#   hotel_name      (str)          "__global__" = no hotel filter applied
#   segment         (str | None)   None = no segment filter applied
#
# Writes to state:
#   retrieved_chunks  (list[dict])  top-k results, each containing:
#     text             (str)   sentence text
#     hotel_name       (str)   source hotel
#     aspect           (str)   matched aspect
#     sentiment        (int)   +1 | -1
#     reviewer_segment (str)   guest segment
#     reviewer_score   (float) original rating
#     similarity_score (float) ChromaDB cosine similarity
# =============================================================================
