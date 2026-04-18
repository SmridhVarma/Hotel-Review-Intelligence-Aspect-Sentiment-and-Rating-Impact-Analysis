# =============================================================================
# summary_retriever.py — Node: Summary Store Retrieval
# =============================================================================
# Purpose:
#   Retrieves the SHAP-based impact summary for the selected hotel (or
#   "__global__" if no hotel is selected) from the ChromaDB summary_store.
#   Used for prioritization queries ("which issue should we fix first?").
#
# Reads from state:
#   hotel_name  (str)  filters summary_store by hotel_name metadata
#
# Writes to state:
#   summary_context  (str | None)  retrieved summary text, None if not found
# =============================================================================
