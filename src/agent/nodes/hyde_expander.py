# =============================================================================
# hyde_expander.py — Node: HyDE Query Expansion
# =============================================================================
# Purpose:
#   Implements Hypothetical Document Embeddings (HyDE). Rather than
#   embedding the raw user query (short, vague), this node prompts GPT-4o
#   to write what an ideal review-based answer would look like, then
#   embeds that hypothetical text using text-embedding-3-small. The
#   resulting embedding is used for ChromaDB similarity search instead of
#   the raw query embedding — producing significantly better semantic matches
#   against actual review text.
#
#   Only runs on "evidence" and "segment" query paths.
#
# Reads from state:
#   query        (str)  current user message
#   hotel_name   (str)  hotel context for hypothetical generation
#
# Writes to state:
#   hyde_embedding  (list[float])  embedding of hypothetical document
#
# Uses: prompts.HYDE_PROMPT, ChatOpenAI (gpt-4o), OpenAIEmbeddings
#       (text-embedding-3-small)
# =============================================================================
