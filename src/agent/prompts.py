# =============================================================================
# prompts.py — LLM Prompt Templates
# =============================================================================
# Purpose:
#   All LangChain prompt templates used by nodes. Keeping prompts here means
#   wording can be iterated without touching node logic or the graph.
#
# Templates (imported by nodes as needed):
#
#   CLASSIFIER_PROMPT
#     Used by : nodes/query_classifier.py
#     Inputs  : {query}, {conversation_history}
#     Task    : Classify query type (evidence | prioritization | mismatch |
#               segment) and extract hotel_name + reviewer segment if present.
#     Returns : JSON string { query_type, hotel_name, segment }
#
#   HYDE_PROMPT
#     Used by : nodes/hyde_expander.py
#     Inputs  : {query}, {hotel_context}
#     Task    : Write a 2–3 sentence hypothetical review excerpt that would
#               ideally answer the query. This is embedded for similarity
#               search — never shown to the user.
#
#   RESPONSE_PROMPT
#     Used by : nodes/response_generator.py  (normal path)
#     Inputs  : {query}, {hotel_context}, {retrieved_context}, {citations}
#     Task    : Answer using only the retrieved context. Cite every claim.
#               Do not assert anything not present in the context.
#
#   CANNOT_ANSWER_PROMPT
#     Used by : nodes/response_generator.py  (low_confidence path)
#     Inputs  : {query}, {hotel_context}
#     Task    : Explain the system cannot answer confidently and suggest
#               rephrasing or narrowing the query.
#
# Tech stack (locked):
#   langchain-core (ChatPromptTemplate)
# =============================================================================
