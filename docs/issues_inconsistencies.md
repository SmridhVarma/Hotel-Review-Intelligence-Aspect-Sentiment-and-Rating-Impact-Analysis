# Issues and Inconsistencies

Known limitations, unresolved questions, and design decisions that involve
statistical or methodological trade-offs. Each entry includes the evidence
behind the concern and the current resolution or open status.

---

## Issue 1: SHAP Reliability for Low-Review Hotels

**Status:** Partially resolved — minimum review threshold applied

**The problem:**

SHAP values are computed by running the global model (trained on all 515k
reviews) on each hotel's review subset. The SHAP values explain the global
model's behavior on that subset — they do not explain what actually drives
ratings for that specific hotel independently.

For hotels with very few reviews, the aspect sentiment features (% positive
per aspect, computed per hotel) are statistically noisy:

- At 50 reviews across 6 aspects → ~8 reviews per aspect on average
- A single additional review flips the sentiment score by ~12 percentage points
- SHAP rankings derived from noisy features amplify this instability

**Data:**

Review count distribution across 1,492 hotels (computed from data.csv):

| Threshold       | Hotels affected | Share of dataset |
|-----------------|----------------|-----------------|
| < 50 reviews    | 158 hotels      | 10.6%           |
| < 100 reviews   | 412 hotels      | 27.6%           |
| < 200 reviews   | 765 hotels      | 51.3%           |
| >= 200 reviews  | 727 hotels      | 48.7%           |

Minimum: 8 reviews (Hotel Gallitzinberg)
Median: 194 reviews
Mean: 345.7 reviews (right-skewed — a few large hotels pull this up)

**Resolution:**

Apply a minimum threshold of **100 reviews** per hotel before computing SHAP.

- Rule of thumb: 10+ observations per feature for stable regression coefficients
- 6 features (aspects) × 10 = 60 minimum; 100 gives a safety margin
- Hotels below 100 reviews receive an `insufficient_data: true` flag in
  summary_store rather than being excluded — the agent surfaces this honestly
  ("This hotel has limited review data; SHAP rankings may not be reliable")

**Open question:**

Whether 100 is the right cutoff depends on Module B's feature engineering —
if aspect scores are computed differently (e.g., TF-IDF weighted rather than
raw counts), the threshold may need adjustment. Revisit after Module B is
implemented.

---

## Issue 2: Multi-Aspect Sentences in Module A

**Status:** Known limitation — flagged in report, not fixed in pipeline

**The problem:**

Module A assigns each sentence to exactly one aspect via keyword/topic
matching. Sentences that naturally span two aspects ("The room was clean but
the staff was rude") are assigned to one aspect only, losing signal for the
other.

Consequence: per-aspect sentiment aggregates in Module B are systematically
underestimated for aspects that frequently co-occur with others in single
sentences. SHAP rankings for such aspects are biased toward understatement.

**Evidence:**

No quantification of co-mention rate is available from the current data.
Hotel review corpora typically have ~15-25% of sentences mentioning multiple
service dimensions (from literature).

**Resolution:**

Not fixed in pipeline — this is a Module A implementation choice and is out
of scope for Module C.

Mitigations applied:
- response_generator system prompt notes that individual review sentences may
  address multiple aspects, so the LLM response acknowledges nuance
- This limitation is noted in the report's limitations section

**Report location:** Add to Section 5 (Limitations) — Module A discussion.
Suggested text: "The sentence-level aspect assignment assumes a one-to-one
mapping between sentences and aspects. Sentences spanning multiple aspects
(e.g., cleanliness and noise mentioned in the same sentence) are assigned
to the dominant keyword match, introducing undercount bias in per-aspect
sentiment aggregates."

---

## Issue 3: __global__ Aggregate Weighting

**Status:** Resolved

**The problem:**

The `__global__` document in summary_store aggregates SHAP values across all
1,492 hotels. A naive unweighted mean gives equal influence to a hotel with
8 reviews and one with 4,789 reviews.

**Resolution:**

Use review-count weighted mean when building the `__global__` aggregate in
`src/agent/ingest.py`:

```python
weight = hotel_review_count / total_reviews
global_shap[aspect] += hotel_shap[aspect] * weight
```

This ensures the global signal reflects data-rich hotels more than sparse ones.

---

## Issue 4: Hotel-to-Hotel Comparison Queries — Out of Scope (v1)

**Status:** Intentionally out of scope — document for future work

**The problem:**

A user asking "compare noise levels at Hotel Arena vs Marriott Paris" creates
an ambiguous routing situation. The query_classifier would set scope=global
(no single hotel), skipping hotel_resolver entirely, but the query actually
references two specific hotels. The current pipeline has no mechanism to
resolve and retrieve evidence for two hotels in a single turn.

**Decision:**

Out of scope for v1. The agent is designed for a single hotel owner querying
their own property — multi-hotel comparison is a different use case with
different retrieval logic (two separate ChromaDB queries, side-by-side
context formatting, comparative response prompting).

**What happens if a user tries it:**

The pipeline defaults to global scope → summary_store → returns the
`__global__` aggregate SHAP doc → response_generator gives a generic
"across all hotels" answer, which is not what the user wanted.
The response will be somewhat useful but not comparative.

**For the report:**

Note this as a v1 limitation in the agent section. Suggested text:
"The current agent is designed for single-property analysis. Queries
comparing multiple hotels are handled as global queries and return
portfolio-level insights rather than property-to-property comparisons."

---

## Report Reminder: Multi-Turn Memory in Conversational Agents Section

**Status:** Add to report when writing the agent section

The report's conversational agent section should explain multi-turn memory —
it is a deliberate architectural decision, not a default Gradio feature.

Key points to cover:

- **Why it matters:** Without memory, follow-up queries ("What about for
  business travelers?") lose hotel and aspect context from the previous turn,
  forcing users to repeat themselves every message. This makes the agent
  unusable for iterative exploration, which is the primary use case.

- **How it works:** LangGraph's `MemorySaver` checkpoints the full `AgentState`
  between invocations, keyed by `thread_id` (one per Gradio session). The
  previous turn's resolved `hotel_name`, `aspects`, and `query_type` persist
  into the next turn automatically — the graph loads the prior state before
  running the first node.

- **Two layers working together:**
  1. `conversation_history` in AgentState — explicit message log passed to
     the query_classifier prompt so the LLM can resolve pronoun references
     and implicit follow-ups in natural language
  2. MemorySaver checkpoint — stores the entire state dict so structured
     fields (hotel_name, aspects, reviewer_segment) persist without the LLM
     needing to re-extract them from the query text

- **Topic shift handling:** If the user mentions a new hotel explicitly,
  `query_classifier` overwrites `hotel_name`. If no hotel is mentioned,
  the previous turn's hotel is kept. This mirrors how a human analyst would
  naturally maintain context in a conversation.

- **Scope for demo:** `MemorySaver` is in-memory only — state is lost on
  server restart. Sufficient for a course demo; production use would require
  `SqliteSaver` or a persistent checkpointer.
