# hyde_expander.py — Stage 5 | Module C (Agent): Generates hypothetical review snippets via GPT-4o and embeds them for ChromaDB retrieval (HyDE).
#
# Input:  AgentState — query, hotel_name, query_direction
# Output: AgentState — hyde_hypotheticals, hyde_embeddings

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from openai import OpenAI

_SRC = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from agent.prompts import HYDE_PROMPT
from agent.state import AgentState

load_dotenv()

EMBED_MODEL = "text-embedding-3-small"

_llm: ChatOpenAI | None = None
_embed_client: OpenAI | None = None


def _get_llm() -> ChatOpenAI:
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(model="gpt-4o", temperature=0.3)
    return _llm


def _get_embed_client() -> OpenAI:
    global _embed_client
    if _embed_client is None:
        _embed_client = OpenAI()
    return _embed_client


def _embed(texts: list[str]) -> list[list[float]]:
    client = _get_embed_client()
    response = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [e.embedding for e in response.data]


def _generate_hypothetical(query: str, hotel_context: str, direction: str) -> str:
    """Generate one hypothetical review sentence synchronously."""
    chain = HYDE_PROMPT | _get_llm()
    result = chain.invoke({
        "query":               query,
        "hotel_context":       hotel_context,
        "sentiment_direction": direction,
    })
    return result.content.strip()


def hyde_expander(state: AgentState) -> dict:
    query         = state["query"]
    hotel_name    = state.get("hotel_name", "__global__")
    direction     = state.get("query_direction", "neutral")
    hotel_context = hotel_name if hotel_name != "__global__" else "all hotels"

    if direction == "neutral":
        # Three hypotheticals for stratified retrieval — generated sequentially
        hypotheticals = [
            _generate_hypothetical(query, hotel_context, "positive"),
            _generate_hypothetical(query, hotel_context, "negative"),
            _generate_hypothetical(query, hotel_context, "neutral"),
        ]
    else:
        hypotheticals = [_generate_hypothetical(query, hotel_context, direction)]

    embeddings = _embed(hypotheticals)

    return {
        "hyde_hypotheticals": hypotheticals,
        "hyde_embeddings":    embeddings,
    }
