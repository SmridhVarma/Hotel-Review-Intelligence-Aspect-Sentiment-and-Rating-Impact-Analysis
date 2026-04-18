# =============================================================================
# response_generator.py — Node: GPT-4o Response Generation
# =============================================================================
# Purpose:
#   Generates the final agent response using GPT-4o. Chooses between two
#   prompt paths based on confidence:
#     - Normal path  (low_confidence=False): RESPONSE_PROMPT — answer the
#       query using retrieved context, cite every factual claim.
#     - Fallback path (low_confidence=True): CANNOT_ANSWER_PROMPT — explain
#       the system cannot answer confidently, suggest rephrasing.
#
#   Formats citations from retrieved_chunks metadata for UI display.
#
# Reads from state:
#   query             (str)        user message
#   hotel_name        (str)        hotel context label
#   retrieved_chunks  (list[dict]) evidence chunks with metadata
#   summary_context   (str | None) SHAP summary text
#   low_confidence    (bool)       selects prompt path
#
# Writes to state:
#   response   (str)        final response text
#   citations  (list[dict]) formatted source metadata for UI:
#                             [ { text, hotel_name, aspect, segment,
#                                 reviewer_score, similarity_score } ]
#
# Uses: prompts.RESPONSE_PROMPT, prompts.CANNOT_ANSWER_PROMPT,
#       ChatOpenAI (gpt-4o)
# =============================================================================
