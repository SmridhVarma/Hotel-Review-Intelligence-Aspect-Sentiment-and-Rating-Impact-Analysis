# prompts.py — Stage 5 | Module C (Agent): LangChain prompt templates for all agent nodes; centralised so wording can be changed without touching node logic.
#
# Input:  None (imported by node files)
# Output: None

from langchain_core.prompts import ChatPromptTemplate

# -----------------------------------------------------------------------------
# CLASSIFIER_PROMPT
# Used by: nodes/query_classifier.py
# Inputs:  {query}, {hotel_name}, {conversation_history}
# Returns: JSON string matching ClassifierOutput schema
# -----------------------------------------------------------------------------
CLASSIFIER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a hotel review analytics assistant. Classify the user query and extract structured fields.

Return a JSON object with exactly these fields:
{{
  "query_type":      "evidence" | "prioritization" | "mismatch",
  "query_direction": "positive" | "negative" | "neutral",
  "aspects":         [],
  "hotel_name":      "<string>",
  "segment":         null | "Business" | "Couple" | "Family" | "Solo" | "Group"
}}

Definitions:
  query_type:
    "evidence"       — asking what guests say about an aspect or experience
    "prioritization" — asking which aspect hurts/helps ratings most, what to fix first
    "mismatch"       — asking about reviews where text sentiment contradicts numeric score

  query_direction:
    "positive" — explicitly asking about good experiences or praise
    "negative" — explicitly asking about complaints, problems, or issues
    "neutral"  — asking for a balanced view, or direction is not specified

  aspects: list of relevant aspects mentioned or implied, from:
    ["Cleanliness", "Staff", "Location", "Noise", "Food", "Room"]
    Leave empty if no specific aspect is mentioned.

  hotel_name:
    Extract from query if a hotel is mentioned.
    If not mentioned, carry forward the current hotel context: {hotel_name}
    Use "__global__" only if the query explicitly asks about all hotels.

  segment: reviewer type if mentioned ("business travellers" → "Business",
    "couples" → "Couple", "families" → "Family", "solo" → "Solo",
    "groups" → "Group"), else null.

Return only the JSON object. No explanation."""),
    ("human", """Conversation so far:
{conversation_history}

Current hotel context: {hotel_name}

Query: {query}"""),
])


# -----------------------------------------------------------------------------
# HYDE_PROMPT
# Used by: nodes/hyde_expander.py
# Inputs:  {query}, {hotel_context}, {sentiment_direction}
# Returns: a 2–3 sentence hypothetical review excerpt
# Notes:   sentiment_direction is "positive", "negative", or "neutral"
#          The result is embedded for similarity search — never shown to user.
# -----------------------------------------------------------------------------
HYDE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are generating a hypothetical hotel review excerpt for semantic search.
Write 2–3 sentences that a real guest would write, matching the sentiment direction given.
The excerpt should be the kind of review text that would ideally answer the query.
Write in first person, past tense. Be specific. Do not invent hotel names or facts.
Return only the excerpt — no explanation, no quotes."""),
    ("human", """Hotel context: {hotel_context}
Query: {query}
Sentiment direction: {sentiment_direction}

Write a hypothetical review excerpt that would answer this query."""),
])


# -----------------------------------------------------------------------------
# RESPONSE_PROMPT
# Used by: nodes/response_generator.py  (normal path)
# Inputs:  {query}, {hotel_context}, {retrieved_context}, {conversation_history}
# Notes:   Agent must cite every factual claim. No hallucination.
# -----------------------------------------------------------------------------
RESPONSE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a hotel analytics assistant helping hotel owners understand guest feedback.

Rules:
- Answer using ONLY the retrieved context provided below.
- Cite every factual claim with [Source N] referencing the numbered sources.
- If the context does not contain enough information to answer, say so clearly.
- Do not assert anything not present in the retrieved context.
- Be concise and direct. Hotel owners want actionable information.
- If asked about a specific aspect, focus on that aspect."""),
    ("human", """Hotel: {hotel_context}
Conversation history: {conversation_history}

Retrieved context:
{retrieved_context}

Question: {query}

Answer with citations:"""),
])


# -----------------------------------------------------------------------------
# CANNOT_ANSWER_PROMPT
# Used by: nodes/response_generator.py  (low_confidence path)
# Inputs:  {query}, {hotel_context}
# Notes:   Triggered when mean retrieval similarity < confidence threshold,
#          or when hotel is unresolved, or when data is insufficient.
# -----------------------------------------------------------------------------
CANNOT_ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a hotel analytics assistant. The retrieval system did not find
sufficiently relevant review content to answer this question confidently.
Explain this briefly and suggest how the user might rephrase or narrow their query.
Be helpful, not apologetic. Keep it to 2–3 sentences."""),
    ("human", """Hotel: {hotel_context}
Question: {query}
Reason for low confidence: {reason}"""),
])


# -----------------------------------------------------------------------------
# SHAP_NARRATIVE_TEMPLATE
# Used by: ingest.py to format summary_store documents
# Not a ChatPromptTemplate — just a string format template.
# -----------------------------------------------------------------------------
SHAP_NARRATIVE_TEMPLATE = (
    "{hotel_name} ({review_count} reviews). "
    "Aspect impact ranking based on global model attribution: "
    "{impact_ranking}. "
    "Positive aspects: {positive_aspects}. "
    "Negative aspects: {negative_aspects}."
)
