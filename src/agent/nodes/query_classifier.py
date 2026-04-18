# =============================================================================
# query_classifier.py — Node: Query Classification
# =============================================================================
# Purpose:
#   First node in the DAG. Classifies the incoming query and extracts
#   any hotel or segment context mentioned in the text, merging with
#   existing state context.
#
# Reads from state:
#   query                (str)       current user message
#   conversation_history (list[dict]) prior turns for context
#   hotel_name           (str)       existing hotel context
#
# Writes to state:
#   query_type  (str)        "evidence"|"prioritization"|"mismatch"|"segment"
#   hotel_name  (str)        updated if hotel mentioned in query
#   last_topic  (str|None)   set to previous query_type before overwriting
#
# Uses: prompts.CLASSIFIER_PROMPT, ChatOpenAI (gpt-4o)
# =============================================================================
