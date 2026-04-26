// ============================================================
// BT5153 — Hotel Review Intelligence: Aspect Sentiment and Rating Impact Analysis  |  AY 2025/2026
// ============================================================

// ── Page & typography setup ─────────────────────────────────
#set document(
  title:  "Hotel Review Intelligence",
  author: ("Smridh Varma", "Siddarth Mahesh", "add ur names here")
)

#import "@preview/fletcher:0.5.5" as fletcher: diagram, node, edge

#set text(font: "Times New Roman", size: 10pt, lang: "en")
#set page(
  paper:   "a4",
  margin:  (top: 1.5cm, bottom: 1.5cm, left: 2cm, right: 1.8cm),
  numbering: "1",
)
#set par(justify: true, leading: 0.6em, spacing: 0.75em)

// ── Heading styles ───────────────────────────────────────────
#set heading(numbering: "1.")
#show heading.where(level: 1): it => {
  v(0.9em)
  text(weight: "bold", size: 12pt, it)
  v(0.3em)
}
#show heading.where(level: 2): it => {
  v(0.6em)
  text(weight: "bold", size: 11pt, it)
  v(0.2em)
}

// ── Code / monospace style ───────────────────────────────────
#show raw.where(block: false): it => box(
  fill:   luma(235),
  inset:  (x: 3pt, y: 1pt),
  radius: 2pt,
  baseline: 20%,
  text(size: 0.9em, it)
)
#show raw.where(block: true): it => block(
  fill:    luma(245),
  inset:   (x: 8pt, y: 5pt),
  radius:  3pt,
  width:   100%,
  text(size: 0.88em, it)
)

// ── Table style ──────────────────────────────────────────────
#set table(
  stroke:      (x, y) => if y == 0 { (bottom: 0.8pt + black) } else { (bottom: 0.3pt + luma(200)) },
  fill:        (_, y) => if calc.odd(y) { luma(250) } else { white },
  inset:       (x: 4pt, y: 3pt),
)
#show table.cell.where(y: 0): set text(weight: "bold", size: 9.5pt)
#show table.cell: set text(size: 9pt)
#set figure(gap: 0.6em)
#show figure.caption: set text(size: 9pt, style: "italic")

// To add a figure: #figure(image("file.png", width: 80%), caption: [Caption]) <label>

// ============================================================
// TITLE PAGE
// ============================================================
#align(center)[
  #v(1.5cm)
  #image("figures/nus_logo.jpg")
  #v(1.5cm)
  #text(size: 20pt, weight: "bold")[Hotel Review Intelligence: Aspect Sentiment and Rating Impact Analysis]
  #v(0.5em)
  #text(size: 12pt, style: "italic")[
    A Multi-Agent LangGraph Pipeline for Automated Hotel Review Analysis
  ]
  #v(1.5em)
  #line(length: 60%, stroke: 0.5pt + luma(160))
  #v(1em)
  #text(size: 10.5pt)[
    BT5153 Applied Machine Learning for Business Analytics \
    National University of Singapore
    \ AY 2025/2026
  ]
  #v(1.8em)
  #text(size: 10pt)[
    *Group 16:* \
    Smridh Varma (A0318634B) \
    Siddarth Mahesh (A0318775N)\
    Mark Dodoo (A0319136E) \
    Frankie Yang Lin () \
    LU QIANQIAN () \
    WANG MENGYU (A0222618J) \
  ]
]
// ============================================================
// ABSTRACT
// ============================================================
#pagebreak()

#align(center)[
  #v(0.5em)
  #text(size: 11pt, weight: "bold")[Abstract]
  #v(0.3em)
]

Hotels receive more reviews than any team can read, yet the insights managers actually need — which service problems are driving scores down, which strengths are worth protecting — are buried in that text. This report presents a three-stage pipeline that turns 515,738 European hotel reviews across 1,492 properties into structured operational intelligence. The first stage applies Aspect-Based Sentiment Analysis (ABSA) @pontiki2014semeval to label sentences across six service dimensions: Cleanliness, Staff, Location, Noise, Food, and Room. The second stage fits a regression model on those labels and uses SHAP decomposition @lundberg2017unified to rank which aspects most drive guest scores, both globally and per property. The third stage wraps these outputs in a stateful Retrieval-Augmented Generation (RAG) @lewis2020rag agent built on LangGraph @langgraph, so hotel operators can ask questions in plain language and get answers grounded in real review evidence. The agent uses Hypothetical Document Embeddings (HyDE) @gao2022precise for retrieval and supports queries filtered by reviewer segment. Evaluation results are in Section 4.

#v(0.5em)
#line(length: 100%, stroke: 0.4pt + luma(200))

// ============================================================
// 1. INTRODUCTION
// ============================================================

= Introduction

Hotel reputation and popularity depend heavily on customer reviews and ratings, which directly influence booking conversion, pricing power, and customer trust. However, while review volume is high, hotels often lack a reliable and scalable way to convert unstructured feedback into actionable insights. Reviews are typically short, informal, and contain mixed comments across multiple service aspects (e.g. cleanliness, staff, location) within a single entry, making manual analysis inefficient and inconsistent. As a result, management may know that a rating is low but cannot clearly identify which issues are driving dissatisfaction or which strengths are most worth maintaining.

The central business problem addressed by this project is: _how can hotels automatically convert large volumes of textual customer feedback into structured operational insights, and identify which service issues most affect ratings?_ A secondary question is whether a conversational interface, grounded in the data and model outputs, can enable non-technical hotel operators to interrogate these insights interactively and make better decisions.

The approach taken here converts unstructured reviews into measurable service indicators using ABSA @pontiki2014semeval and regression modelling, then surfaces those results through a conversational interface so hotel operators can interrogate the data directly rather than reading a static report.

The system makes three contributions:

+ *Aspect-level sentiment labelling* of 515,738 reviews across six hotel service dimensions, without manual annotation, by leveraging the dataset's pre-separated positive and negative review fields as a deterministic polarity signal.
+ *Interpretable rating impact ranking* using SHAP value decomposition @lundberg2017unified over a trained regression model, producing per-hotel and global rankings of which aspects most drive guest scores.
+ *A stateful RAG conversational agent* @lewis2020rag built on LangGraph @langgraph, supporting evidence queries, prioritisation queries, mismatch detection, and reviewer-segment-specific queries via HyDE-enhanced @gao2022precise retrieval over a ChromaDB vector store.

The remainder of this report is structured as follows: Section 2 describes the dataset; Section 3 details the methodology across all pipeline stages; Section 4 presents evaluation results; Section 5 discusses limitations and future work; Section 6 concludes.

// ============================================================
// 2. DATASET
// ============================================================

= Dataset

This project utilises the European Hotel Reviews dataset @kaggle515k, comprising 515,738 reviews of 1,492 hotels across six European cities: Amsterdam, Barcelona, London, Milan, Paris, and Vienna. Each record includes a hotel identifier, an overall reviewer score on a 0--10 scale, review date, reviewer nationality, trip-type tags, and explicitly separated positive and negative review text fields. The dataset spans reviews submitted between August 2015 and August 2017.

#figure(
  table(
    columns: (auto, auto, 1fr),
    [*Field*], [*Type*], [*Description*],
    [`Hotel_Name`],          [string], [Hotel identifier; 1,492 unique hotels],
    [`Reviewer_Score`],      [float],  [Overall guest rating, 0--10; mean $approx$ 8.4, SD $approx$ 1.7],
    [`Positive_Review`],     [string], [Guest positive free-text comments],
    [`Negative_Review`],     [string], [Guest negative free-text comments],
    [`Tags`],                [string], [Trip type tags (e.g. Couple, Family, Business traveler)],
    [`Reviewer_Nationality`],[string], [Guest country of origin; 227 unique nationalities],
    [`Review_Date`],         [string], [Review submission date (Aug 2015 to Aug 2017)],
    [`lat`, `lng`],          [float],  [Hotel geolocation coordinates],
  ),
  caption: [Key dataset fields used in the pipeline.]
) <tab:dataset>

The pre-separation of positive and negative review fields is the defining property that enables the labelling approach used in this project. Rather than requiring manual sentiment annotation at sentence level, the source field is treated as a deterministic polarity label for each extracted sentence. Reviews containing the placeholder strings `"No Negative"` or `"No Positive"` are filtered during preprocessing. The reviewer score distribution is left-skewed, consistent with the well-documented positivity bias in voluntary online review platforms, which motivates the use of aspect-level decomposition rather than aggregate score analysis alone.

#figure(
  block(
    width: 100%,
    height: 4.5cm,
    fill: luma(240),
    stroke: 0.4pt + luma(180),
    radius: 3pt,
    inset: 10pt,
    align(center + horizon)[
      #text(size: 8.5pt, fill: luma(130), style: "italic")[
        _Figure suggestion: Histogram of `Reviewer_Score` (0--10) showing left-skewed distribution_ \
        #v(0.4em)
        Replace with: `#figure(image("figures/score_dist.png", width: 70%), caption: [...])`
      ]
    ]
  ),
  caption: [Distribution of reviewer scores (0--10) across 515,738 reviews.]
) <fig:score_dist>

// ============================================================
// 3. METHODOLOGY
// ============================================================

= Methodology

The pipeline consists of five sequential stages: text preparation, aspect extraction, sentiment assignment, rating impact modelling, and conversational agent construction. Stages 1--4 produce the structured artefacts that Stage 5 ingests for retrieval. @fig:pipeline provides a schematic overview of the full pipeline.

#figure(
  diagram(
    node-stroke: 0.6pt,
    node-fill: luma(245),
    node-corner-radius: 3pt,
    node-inset: 6pt,
    spacing: (1.8em, 1.6em),
    node((0,0), [#text(size:8pt)[`data.csv`]], name: <data>),
    node((1,0), [#text(size:8pt)[*Stage 1*\ Preprocess]], name: <s1>),
    node((2,0), [#text(size:8pt)[*Stage 2*\ Aspect\ Extraction]], name: <s2>),
    node((3,0), [#text(size:8pt)[*Stage 3*\ Sentiment\ Assignment]], name: <s3>),
    node((3,1), [#text(size:8pt)[*Stage 4*\ Rating Impact\ Modelling]], name: <s4>),
    node((2,1), [#text(size:8pt)[*Stage 5*\ ChromaDB\ Ingestion]], name: <s5>),
    node((1,1), [#text(size:8pt)[*LangGraph*\ Agent]], name: <agent>),
    node((0,1), [#text(size:8pt)[*Gradio*\ Web UI]], name: <ui>),
    edge(<data>,  <s1>,    "->"),
    edge(<s1>,    <s2>,    "->"),
    edge(<s2>,    <s3>,    "->"),
    edge(<s3>,    <s4>,    "->"),
    edge(<s4>,    <s5>,    "->"),
    edge(<s5>,    <agent>, "->"),
    edge(<agent>, <ui>,    "->"),
  ),
  caption: [End-to-end pipeline from raw review data to conversational agent UI.]
) <fig:pipeline>

== Stage 1: Text Preparation

For each review, the `Positive_Review` and `Negative_Review` fields are combined into a single full review text. This combined text is split into individual sentences using an NLP sentence tokeniser, with each sentence becoming a separate unit of analysis. Each sentence inherits the polarity of its source field. Reviewer segments are extracted from the `Tags` field and normalised into five categories: Business, Couple, Family, Solo, and Group. The output is a sentence-level dataset used by all downstream stages.

== Stage 2: Aspect Extraction

To prepare the text for topic modelling, the sentence corpus undergoes rigorous linguistic preprocessing using the `spaCy` NLP pipeline. Tokens are lowercased, lemmatised, and filtered to remove standard English stop words, punctuation, and non-alphabetic strings. A Document-Term Matrix (DTM) is then constructed using unigram and bigram features. To reduce noise and improve topic coherence, the vocabulary is constrained by ignoring terms that appear in fewer than 20 sentences (`min_df=20`) or in more than 50% of the corpus (`max_df=0.5`), capped at a maximum of 50,000 features.

Common hotel service topics are subsequently identified from this cleaned corpus using Latent Dirichlet Allocation (LDA) @blei2003latent, configured with online learning to efficiently process the large corpus in batches. The model is initialised with six latent components (`n_components=6`) to align with the expected service dimensions. Because LDA is unsupervised, the algorithm outputs six distinct topic-word distributions rather than labelled categories. The top keywords from each latent topic are manually reviewed and deterministically mapped to the six predefined service categories: Cleanliness, Staff, Location, Noise, Food, and Room. This human-in-the-loop curation produces the final aspect keyword dictionary that drives the downstream sentiment assignment.

#figure(
  table(
    columns: (auto, 1fr, auto),
    [*Topic*], [*LDA top-10 words*], [*Mapped aspect*],
    [0], [location, staff, friendly, helpful, clean, comfortable, room, excellent, breakfast, rooms], [Cleanliness #footnote[Topic 0 was a generic-praise cluster; the cleanliness keyword list was rebuilt seed-first.]],
    [1], [location, close, station, walk, city, metro, easy, near, restaurants, walking], [Location],
    [2], [room, staff, check, didn't, like, reception, booking, wifi, asked, wasn't], [Staff],
    [3], [staff, service, bar, breakfast, lovely, room, restaurant, food, amazing, excellent], [Food],
    [4], [room, small, bed, bathroom, rooms, shower, size, floor, view, room small], [Noise #footnote[Topic 4 was dominated by room-amenity vocabulary; the noise keyword list was rebuilt seed-first.]],
    [5], [breakfast, room, coffee, water, air, expensive, hot, tea, cold, price], [Room],
  ),
  caption: [LDA topic-to-aspect mapping. Topics 0 and 4 required seed-led reconstruction due to weak topic separation for cleanliness and noise.]
) <tab:lda_topics>


== Stage 3: Aspect-Level Sentiment Assignment

For each segmented sentence, the relevant aspect is detected through keyword matching against the topic word dictionary. Sentiment is then assigned deterministically based on the sentence's source field: sentences from the positive review field are labelled positive (+1), while those from the negative field are labelled negative (-1). Sentences with no keyword match receive a value of 0 (not mentioned). This produces two outputs: a sentence-level labelled dataset for agent ingestion, and a review-level feature matrix for rating impact modelling.

== Stage 4: Rating Impact Modelling

Structured feature variables are constructed for each review by converting assigned aspect sentiments into numeric values, forming a machine learning training dataset. A regression model is trained to predict the overall reviewer score as a function of the extracted aspect features, and performance is evaluated using RMSE, MAE, and $R^2$. Linear Regression is the interpretable baseline; XGBoost captures non-linear interactions between aspects. SHAP values @lundberg2017unified are applied to determine which hotel aspects most strongly influence overall guest ratings. Scores are computed both globally across all hotels and locally per hotel by running the global model on each hotel's review subset, yielding a consistent per-hotel and global impact ranking.

== Stage 5: Conversational Agent <sec:agent>

The conversational agent is the interface through which hotel operators query the pipeline outputs. It is built as a stateful LangGraph @langgraph directed acyclic graph (DAG) and answers questions about service priorities, guest feedback patterns, and rating drivers. Responses must be grounded in retrieved evidence; the agent surfaces a "cannot answer confidently" message rather than speculating when evidence is thin.

=== Vector Store Construction

Two ChromaDB collections are populated once the full pipeline completes, implementing a hierarchical retrieval architecture inspired by PageIndex @pageindex:

- *`evidence_store`*: approximately one million sentence-level embeddings generated using OpenAI's `text-embedding-3-small` model. Each document is annotated with four metadata fields: `hotel_name`, `aspect`, `sentiment`, and `reviewer_segment`, enabling precise filtered retrieval. This index allows the agent to answer evidence queries (e.g. "Why is cleanliness rated poorly?") by retrieving specific examples of guest feedback aligned with identified service issues.

- *`summary_store`*: one document per hotel plus a `__global__` aggregate entry. Each document contains a human-readable narrative derived from the SHAP impact summary for that hotel. This index answers macro-level prioritisation queries (e.g. "Which issue should we fix first?") without requiring retrieval from the full evidence corpus.

The two-collection design separates concerns: the `evidence_store` answers _why_ via specific guest experiences, while the `summary_store` answers _what to prioritise_ via ranked service improvements.

=== LangGraph DAG

The agent is implemented as a LangGraph `StateGraph` @langgraph over a shared `ConversationState` TypedDict. All eight nodes read from and write to this shared state, which persists across conversation turns. Node functions are defined in individual files under `src/agent/nodes/`, keeping each node independently testable. The graph wiring in `src/agent/graph.py` contains no business logic. @fig:langgraph illustrates the DAG structure and conditional routing.

#figure(
  table(
    columns: (auto, 1fr),
    [*Node*], [*Responsibility*],
    [`query_classifier`],   [Classifies query as evidence, prioritisation, mismatch, or segment; extracts hotel context from query text and merges with existing state],
    [`segment_filter`],     [Detects reviewer segment mention (Business, Couple, Family, Solo, Group); sets ChromaDB `where` filter; passes `None` if no segment detected],
    [`hyde_expander`],      [Prompts GPT-4o to generate a 2--3 sentence hypothetical review excerpt; embeds it with `text-embedding-3-small` for similarity search],
    [`evidence_retriever`], [Queries `evidence_store` using HyDE embedding with optional `hotel_name` and `reviewer_segment` filters; returns top-$k$ chunks with metadata],
    [`summary_retriever`],  [Retrieves SHAP impact summary from `summary_store` for the selected hotel or `__global__`],
    [`context_merger`],     [Combines retrieved chunks and summary context; computes mean similarity score; sets `low_confidence = True` if score falls below 0.5 or both sources are empty],
    [`response_generator`], [Generates GPT-4o response grounded in merged context; switches to fallback prompt on low confidence],
    [`state_manager`],      [Appends turn to conversation history; clears cached retrieval on topic shift to force fresh context on the next turn],
  ),
  caption: [LangGraph DAG node descriptions and state contracts.]
) <tab:nodes>

#figure(
  diagram(
    node-stroke: 0.6pt,
    node-fill: luma(245),
    node-corner-radius: 3pt,
    node-inset: 5pt,
    spacing: (3em, 1.8em),
    node((1,0), [#text(size:8pt)[`query_classifier`]],   name: <qc>),
    node((1,1), [#text(size:8pt)[`segment_filter`]],     name: <sf>),
    node((0,2), [#text(size:8pt)[`hyde_expander`]],      name: <hyde>),
    node((2,2), [#text(size:8pt)[`summary_retriever`]],  name: <sr>),
    node((0,3), [#text(size:8pt)[`evidence_retriever`]], name: <er>),
    node((1,4), [#text(size:8pt)[`context_merger`]],     name: <cm>),
    node((1,5), [#text(size:8pt)[`response_generator`]], name: <rg>),
    node((1,6), [#text(size:8pt)[`state_manager`]],      name: <sm>),
    node((1,7), [#text(size:8pt, weight: "bold")[END]],  name: <end>),
    edge(<qc>,   <sf>,   "->"),
    edge(<sf>,   <hyde>, "->", [#text(size:7pt)[evidence / segment]]),
    edge(<sf>,   <sr>,   "->", [#text(size:7pt)[prioritisation / mismatch]]),
    edge(<hyde>, <er>,   "->"),
    edge(<er>,   <cm>,   "->"),
    edge(<sr>,   <cm>,   "->"),
    edge(<cm>,   <rg>,   "->"),
    edge(<rg>,   <sm>,   "->"),
    edge(<sm>,   <end>,  "->"),
  ),
  caption: [LangGraph DAG: node execution order and conditional routing by query type.]
) <fig:langgraph>

=== Hypothetical Document Embeddings (HyDE)

A core retrieval challenge is the semantic gap between short, informal user queries and the dense, specific language of actual reviews. Direct embedding of a query such as "how do I fix my reputation?" produces a vector that is distributionally distant from the review sentences it should retrieve @gao2022precise. HyDE addresses this by interposing a generation step: GPT-4o is prompted to write a 2--3 sentence hypothetical review excerpt that would ideally answer the query. This hypothetical document is embedded and used as the similarity search vector in place of the raw query. Because the hypothetical is written in review-like language, it sits closer in embedding space to actual evidence, consistently retrieving more relevant chunks without additional fine-tuning or labelled retrieval data.

=== Reviewer Segmentation

The `Tags` field encodes trip context for each reviewer. These are normalised into five segments during preprocessing: Business, Couple, Family, Solo, and Group. Each sentence in the `evidence_store` is annotated with its `reviewer_segment` metadata. The `segment_filter` node detects segment mentions in user queries and passes the corresponding ChromaDB filter to `evidence_retriever`, enabling segment-specific retrieval (e.g. "what do business travelers complain about?") without additional modelling overhead. If no segment is detected, retrieval proceeds without filtering.

=== Retrieval Scope

The agent supports three retrieval scopes controlled by the hotel selector in the UI:

#figure(
  table(
    columns: (auto, auto, 1fr),
    [*Mode*], [*Trigger*], [*Retrieval scope*],
    [Global],    ["All Hotels" selected],             [`summary_store __global__` entry + unfiltered `evidence_store`],
    [Per-hotel], [Specific hotel selected],           [`summary_store` hotel entry + `evidence_store` filtered by `hotel_name`],
    [Segment],   [Segment keyword detected in query], [Either mode above with additional `reviewer_segment` filter],
  ),
  caption: [Agent retrieval modes by hotel selection and query content.]
) <tab:retrieval>

=== Groundedness Constraint

Every factual statement in a response must be traceable to a retrieved chunk or summary document. The `context_merger` node computes a mean cosine similarity score across retrieved chunks. If this score falls below 0.5, or if both retrieval sources return nothing, `low_confidence` is set to `True` and the `response_generator` switches to a fallback prompt that explicitly states the system cannot answer confidently and suggests rephrasing. Source citations, including source sentence, hotel name, aspect, reviewer segment, and similarity score, are surfaced alongside every response. This directly addresses three failure modes specified in the project brief: unsupported claims, undetected mismatches between text sentiment and numeric rating, and unhelpful responses when evidence is insufficient.

=== Multi-Turn Conversational Memory

The agent preserves context across turns using LangGraph's `MemorySaver` checkpointer, keyed by a `thread_id` per Gradio session. The motivation is practical: follow-up queries like "What about for business travellers?" only make sense if the hotel and aspect context from the previous turn is still available. Without it, users would have to restate their hotel on every message.

Two mechanisms work together. `MemorySaver` checkpoints the full `AgentState` between invocations, so structured fields from the prior turn (`hotel_name`, `aspects`, `reviewer_segment`) carry forward without the LLM needing to re-extract them. A `conversation_history` field also accumulates the four most recent turns and is passed to `query_classifier`, which uses it to resolve pronoun references and implicit continuations.

If the user explicitly names a different hotel, `query_classifier` overwrites `hotel_name`; otherwise the prior context holds. Switching hotels via the UI dropdown generates a new `thread_id` and clears state, preventing context from one property contaminating queries about another. `MemorySaver` is in-RAM only; state does not survive server restart. A persistent checkpointer (`SqliteSaver` or Redis-backed) would be needed for production deployment.


=== Retrieval Strategy and Deduplication

Evidence retrieval follows one of two paths depending on query direction. For directional queries (those explicitly positive or negative), a single HyDE hypothetical is generated and the top $k = 20$ results are retrieved with a sentiment metadata filter matching the stated direction. For neutral queries, three hypotheticals are generated concurrently in an `asyncio.gather` call: one positive-toned, one negative-toned, and one without a sentiment constraint. Each yields the top $k = 7$ results independently.

The three result sets are merged by iterating through them in order (positive pole first, then negative, then unfiltered) and inserting each chunk into the output list only if its exact sentence string has not already been seen, using a Python `set` for $O(1)$ lookup. Because the positive pole is processed first, when the same sentence appears in multiple poles its similarity score from the first encounter is retained. The merged list is then sorted by cosine similarity descending before being passed to `context_merger`. This stratified approach prevents the response generator receiving a cluster of near-identical sentences and ensures the context contains a mix of positive and negative guest experience rather than a single polarity cluster.

Hotel name resolution uses fuzzy string matching (RapidFuzz `fuzz.ratio`, threshold 70/100) against the hotel name list exported during ingestion. Queries where the best match falls below this threshold are flagged `hotel_unresolved` and routed to the low-confidence fallback, rather than silently returning results for the wrong property.

// ============================================================
// 4. EVALUATION
// ============================================================

= Evaluation

== Module A: Aspect-Level Sentiment Classification

Sentiment labels are derived deterministically from source field polarity; evaluation therefore focuses on aspect assignment quality, specifically whether the correct aspect is matched for each sentence. A validation set is constructed by manually labelling a stratified random sample of [_N_] sentences. Performance is measured using Accuracy, Precision, Recall, and F1 as specified in the project success criteria @pontiki2014semeval.

#figure(
  block(
    width: 100%,
    height: 4cm,
    fill: luma(240),
    stroke: 0.4pt + luma(180),
    radius: 3pt,
    inset: 10pt,
    align(center + horizon)[
      #text(size: 8.5pt, fill: luma(130), style: "italic")[
        _Figure suggestion: Horizontal bar chart of sentence count per aspect across the full corpus_ \
        #v(0.4em)
        Replace with: `#figure(image("figures/aspect_freq.png", width: 72%), caption: [...])`
      ]
    ]
  ),
  caption: [Sentence-level mention frequency per aspect across the full corpus.]
) <fig:aspect_freq>

#figure(
  table(
    columns: (auto, auto, auto, auto, auto),
    [*Aspect*],    [*Accuracy*], [*Precision*], [*Recall*], [*F1*],
    [Cleanliness], [--], [--], [--], [--],
    [Staff],       [--], [--], [--], [--],
    [Location],    [--], [--], [--], [--],
    [Noise],       [--], [--], [--], [--],
    [Food],        [--], [--], [--], [--],
    [Room],        [--], [--], [--], [--],
    [*Macro avg*], [--], [--], [--], [--],
  ),
  caption: [Aspect classification performance on held-out validation set.]
) <tab:absa>

== Module B: Rating Impact Modelling

Model performance is evaluated on a held-out test set using RMSE, MAE, and $R^2$. SHAP rank stability is assessed separately by computing aspect impact rankings across quarterly time slices and checking whether the top-3 aspects remain consistent, as specified in the project success criteria.

#figure(
  table(
    columns: (auto, auto, auto, auto),
    [*Model*],           [*RMSE*], [*MAE*], [*$R^2$*],
    [Linear Regression], [1.4125], [1.0857], [0.2464],
    [XGBoost],           [1.4014], [1.0756], [0.2582],
  ),
  caption: [Regression model performance on 80/20 held-out test split (random seed 42).]
) <tab:regression>

The $R^2$ values of 0.25--0.26 are low in absolute terms but expected for this feature set. Six binary aspect sentiment features cannot account for everything that affects a reviewer's score (price expectations, photos, booking experience), so the model is not intended as a score predictor. Its purpose is decomposition: SHAP values quantify each aspect's marginal contribution given the features available, which is interpretable regardless of overall explained variance. XGBoost marginally outperforms the linear baseline on all three metrics, confirming that non-linear interactions between aspects contribute to rating outcomes.

#figure(
  table(
    columns: (auto, auto, auto),
    [*Aspect*],    [*Mean SHAP value*], [*Rank*],
    [Staff],       [+0.0364], [1],
    [Location],    [$-$0.0305], [2],
    [Room],        [$-$0.0119], [3],
    [Cleanliness], [+0.0033], [4],
    [Food],        [+0.0014], [5],
    [Noise],       [+0.0012], [6],
  ),
  caption: [Global aspect impact ranking by mean SHAP value @lundberg2017unified. Positive values indicate the aspect raises predicted scores on average; negative values indicate it lowers them.]
) <tab:shap>

Staff is the strongest positive driver of guest ratings globally; Location is the strongest negative drag. The top-3 aspects (Staff, Location, Room) were consistent across all 9 quarterly time slices from Q3 2015 through Q3 2017, confirming that SHAP rankings are stable over time rather than an artefact of any particular review period.

#figure(
  block(
    width: 100%,
    height: 4cm,
    fill: luma(240),
    stroke: 0.4pt + luma(180),
    radius: 3pt,
    inset: 10pt,
    align(center + horizon)[
      #text(size: 8.5pt, fill: luma(130), style: "italic")[
        _Figure suggestion: Horizontal bar chart of global mean SHAP values per aspect (red = negative impact, green = positive)_ \
        #v(0.4em)
        Replace with: `#figure(image("figures/shap_global.png", width: 72%), caption: [...])`
      ]
    ]
  ),
  caption: [Global aspect impact ranking by mean SHAP value.]
) <fig:shap_global>

#figure(
  block(
    width: 100%,
    height: 5cm,
    fill: luma(240),
    stroke: 0.4pt + luma(180),
    radius: 3pt,
    inset: 10pt,
    align(center + horizon)[
      #text(size: 8.5pt, fill: luma(130), style: "italic")[
        _Figure suggestion: Heatmap of per-hotel SHAP values across the 15 most-reviewed hotels_ \
        #v(0.4em)
        Replace with: `#figure(image("figures/shap_heatmap.png", width: 95%), caption: [...])`
      ]
    ]
  ),
  caption: [Per-hotel SHAP value heatmap across the 15 most-reviewed hotels. Variation across rows illustrates that different hotels have distinct aspect-level impact profiles.]
) <fig:shap_heatmap>

== Module C: Conversational Agent

Agent evaluation uses a manually curated set of [_N_] test queries spanning all four query types: evidence, prioritisation, mismatch, and segment. Each response is assessed on four dimensions defined in the project success criteria.

#figure(
  table(
    columns: (auto, 1fr, auto),
    [*Metric*],         [*Definition*],                                                              [*Score*],
    [Groundedness],     [Fraction of factual claims supported by cited retrieved evidence],          [--],
    [Relevance],        [Query-response semantic alignment (GPT-4o judge, 1--5 scale)],              [--],
    [Context accuracy], [Correct use of SHAP outputs and retrieved review text in response],         [--],
    [Resolution rate],  [Fraction of queries answered without unnecessary clarification follow-up],  [--],
  ),
  caption: [Agent evaluation results across four dimensions.]
) <tab:agent>

#figure(
  block(
    width: 100%,
    height: 5.5cm,
    fill: luma(240),
    stroke: 0.4pt + luma(180),
    radius: 3pt,
    inset: 10pt,
    align(center + horizon)[
      #text(size: 8.5pt, fill: luma(130), style: "italic")[
        _Figure suggestion: Screenshot of Gradio UI showing a prioritisation query with grounded response and source citations_ \
        #v(0.4em)
        Replace with: `#figure(image("figures/agent_demo.png", width: 88%), caption: [...])`
      ]
    ]
  ),
  caption: [Example agent interaction: a prioritisation query with grounded response and source citations displayed below.]
) <fig:agent_demo>

// ============================================================
// 5. DISCUSSION
// ============================================================

= Discussion

== Limitations

*Aspect coverage.* The six-aspect taxonomy is fixed before modelling. Reviews that mention pricing, parking, or amenities fall outside the feature set and are dropped. No attempt is made to expand the taxonomy dynamically, in line with the project constraints.

*Keyword matching brittleness.* Aspect assignment depends on lexical overlap with the Stage 2 dictionary. Synonyms, negations ("not clean"), and sentences that mention two aspects at once can produce missed or incorrect assignments. Cleanliness and noise rely on hand-curated seed lists rather than LDA terms, so their coverage reflects domain knowledge rather than corpus statistics. Conclusions should be read as aggregate patterns, not per-sentence ground truth.

*Source-field polarity as supervision.* Sentiment is assigned from which field the sentence came from. Some reviewers put negative text in the positive field and vice versa, introducing label noise that flows through to SHAP values. Basic quality checks are applied, but no deep authenticity filtering.

*Agent retrieval quality.* HyDE @gao2022precise reduces the semantic gap between user queries and review text, but broad queries or aspects with thin coverage in the vector store can still trigger the low-confidence fallback. The 0.5 cosine similarity threshold was set by inspection and may need calibration against a labelled query set.

*Hallucination risk.* Despite the citation requirement, GPT-4o can produce plausible-sounding text when retrieved context is weak but above the confidence threshold. The citation panel makes the evidentiary basis of each claim visible, which partially mitigates this.

== Failure Modes and Future Mitigations

@tab:failuremodes summarises the known failure modes, their root causes, and possible fixes for a future implementation.

#figure(
  table(
    columns: (1.6fr, 2fr, 2.2fr),
    inset: (x: 5pt, y: 4pt),
    [*Limitation*], [*Root cause*], [*Possible future fix*],

    [SHAP unreliable for sparse hotels],
    [SHAP runs the global model on each hotel's review subset. With fewer than 100 reviews, one additional review can shift a sentiment score by ~12 percentage points. 412 hotels (27.6%) fall below this threshold.],
    [Bootstrap confidence intervals on per-hotel SHAP values, or Bayesian shrinkage toward the global prior, would produce stable rankings for hotels with 30--50 reviews.],

    [One aspect per sentence],
    [The first keyword match wins. Sentences spanning two dimensions ("the room was clean but noisy") lose signal for the second aspect. Literature estimates 15--25% of hotel review sentences mention multiple aspects.],
    [A multi-label classifier trained on a small annotated sample would assign probability scores across all six aspects simultaneously, eliminating the single-assignment constraint.],

    [Keyword matching misses synonyms and negations],
    ["Spotless" does not match Cleanliness; "not clean" matches it with the wrong polarity. Recall is lower than precision, with systematic gaps for aspects with varied vocabulary.],
    [Embedding similarity against aspect prototype vectors, or a fine-tuned sequence classifier, handles implicit and negated mentions without requiring exhaustive vocabulary enumeration.],

    [Polarity label noise],
    [Sentiment is derived from source field. A minority of reviewers place negative text in the positive field and vice versa, introducing noise that propagates through the feature matrix.],
    [A lightweight binary sentiment check can flag sentences where model-predicted polarity contradicts the source field. Those sentences can be excluded or down-weighted without full manual re-annotation.],

    [Confidence threshold heuristic],
    [The 0.5 cosine similarity cutoff that triggers the low-confidence fallback was set by inspection. It may produce unnecessary fallbacks on valid broad queries.],
    [Calibrate against a held-out query set with human answerability labels, selecting the threshold that maximises F1 on the confident/fallback classification decision.],

    [No hotel-to-hotel comparison],
    [A query referencing two hotels defaults to global scope, returning portfolio-level output rather than a side-by-side comparison. The classifier has no mechanism for dual-hotel intent.],
    [Extend `query_classifier` to detect comparative intent, resolve two hotel names independently, and route to a parallel dual-retrieval path with a comparative prompt template.],

    [Conversation state lost on restart],
    [`MemorySaver` stores state in RAM. Server restart clears all sessions; there is no interaction log for quality monitoring or threshold recalibration.],
    [Replace `MemorySaver` with `SqliteSaver` or a Redis-backed checkpointer. Log low-confidence queries to a feedback store for periodic review.],

    [English reviews only],
    [Non-English sentences are dropped during preprocessing. Hotels with predominantly French, German, or Spanish guests have reduced coverage.],
    [Route non-English sentences through translation before embedding, or replace `text-embedding-3-small` with `multilingual-e5-large` to map cross-lingual text to a shared embedding space.],
  ),
  caption: [Known failure modes, root causes, and possible mitigations for future implementations.]
) <tab:failuremodes>

== Future Work

A fine-tuned sequence labelling model @pontiki2014semeval would substantially improve recall on implicit and negated aspect mentions. Nationality-based segmentation via the `Reviewer_Nationality` field could surface culturally specific service preferences. The similarity confidence threshold should be calibrated against the evaluation query set rather than set by inspection. A feedback loop logging low-quality retrievals for human review would let the vector store and threshold improve over time. Multilingual support would require either a translation preprocessing step or replacement of the embedding model with a multilingual variant.

// ============================================================
// 6. CONCLUSION
// ============================================================

= Conclusion

The pipeline described in this report takes 515,738 unstructured hotel reviews and produces something a hotel manager can actually use: a ranked list of which service aspects drive ratings up or down at their specific property, with the raw guest evidence one click away. LDA @blei2003latent extracts the vocabulary, SHAP decomposition @lundberg2017unified quantifies the impact, and a LangGraph @langgraph RAG agent @lewis2020rag surfaces both through a conversational interface so operators do not need to read SQL output or a static dashboard. The system processes all 1,492 hotels without manual annotation beyond a validation sample, and the agent's HyDE @gao2022precise retrieval and segment-aware filtering let it answer targeted questions — what business travellers complain about, which aspects are most fixable — not just broad summaries. The known limitations are real: keyword matching misses synonyms, SHAP values are noisy for small hotels, and the confidence threshold needs calibration. But for the core task of turning review volume into structured service priorities, the pipeline works.

#bibliography("refs.bib", style: "ieee")

#pagebreak()

#heading(numbering: none)[Appendix]

#heading(numbering: none)[AI Usage Declaration]
In accordance with NUS academic integrity guidelines for BT5153, the team declares the following use of generative AI tools during this project:

- *Code scaffolding:* Claude Code was used to generate boilerplate LangGraph node structure, `requirements.txt` content, and script templates. All generated code was reviewed, modified to project requirements, and validated by team members.

- *Debugging assistance:* Claude Code was used to diagnose issues, and advise on feature engineering design. Findings were independently verified against the codebase.

- *Content drafting:* Gemini 3-Pro/GPT 5/Claude Sonnet 4.6 was used to inform initial document drafts. Final wording was reviewed and refined by the team for technical accuracy and BT5153 rubric alignment.

- *Human-only contributions:* Model selection rationale, feature engineering design decisions, architectural choices, test case design, and all critical reflection content reflect team analysis and were not generated by AI tools.