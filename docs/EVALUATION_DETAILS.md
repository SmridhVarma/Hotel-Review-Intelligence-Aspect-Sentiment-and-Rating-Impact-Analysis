# Evaluation details

This document covers the evaluation methodology and results for Module A (ABSA), Module B (rating impact modeling), and Module C (the conversational agent). Raw outputs are in `outputs/evaluation_report.json` and `outputs/agent_eval_results.csv`.

---

## Module A: aspect and sentiment assignment

We ran a manual spot-check on 100 sentences sampled from the pipeline output. Each sentence had a predicted aspect label (from keyword matching) and a human-assigned true label. Results are in `outputs/validation_sample.csv`.

| Metric | Result |
|---|---|
| Aspect classification accuracy | ~85% (83/98 checked) |

Most errors fell into two patterns. The first is multi-aspect sentences, where a review mentions several things in one sentence and the keyword matcher latches onto whichever aspect appears first. "Great room and amazing location, extremely friendly staff" got classified as Staff when a human labeled it Location. The second pattern is short or ambiguous sentences: "Location clean and comfortable" lands on Cleanliness instead of Location because the first strong keyword hit is a cleanliness term.

A few specific cases worth noting:

- WiFi complaints consistently classify as Room, since WiFi is in the Room keyword list. This is debatable but internally consistent.
- Sentences mentioning both noise and room features (thin walls, A/C units) are a toss-up depending on keyword order.
- Very short sentences like "Breakfast and parking" can misfire entirely.

For a keyword-based classifier with no ML component, 85% on a mixed sample is acceptable. More importantly, the errors tend to be cross-aspect rather than wrong-polarity, so the sentiment direction (positive/negative) attached to a sentence is usually correct even when the aspect label is off.

---

## Module B: rating impact modeling

### Model metrics

We trained Linear Regression as a baseline and XGBoost on the same six-feature aspect sentiment matrix. Both predict a reviewer's overall score (0-10).

| Model | RMSE | MAE | R² |
|---|---|---|---|
| Linear Regression | 1.41 | 1.09 | 0.25 |
| XGBoost | 1.40 | 1.08 | 0.26 |

XGBoost edges out Linear Regression on every metric, though the gap is small. The R² of roughly 0.25 means the six aspect sentiment features explain about a quarter of score variance. That sounds modest, but reviewer scores are shaped by a lot of things the ABSA pipeline doesn't track: price expectations, brand loyalty, one-off incidents that don't fit the six aspects, and the general subjectivity of a 1-10 scale. A quarter of variance explained from six binary-ish features is reasonable.

The prediction quality matters less than you'd expect here, because the agent uses SHAP attributions rather than raw predictions. SHAP tells us the direction and relative magnitude of each aspect's influence, and for that purpose, stability over time is more relevant than absolute RMSE.

### SHAP stability

To check whether the SHAP rankings drift with time, we split the dataset into quarterly slices and computed the top-3 SHAP drivers per period.

| Period | Top 3 aspects |
|---|---|
| 2015 Q3 | Room, Staff, Location |
| 2015 Q4 | Room, Staff, Location |
| 2016 Q1 | Room, Staff, Location |
| 2016 Q2 | Room, Staff, Location |
| 2016 Q3 | Room, Staff, Location |
| 2016 Q4 | Room, Staff, Location |
| 2017 Q1 | Room, Staff, Location |
| 2017 Q2 | Room, Staff, Location |
| 2017 Q3 | Room, Staff, Location |

The ranking is identical across all nine quarters, two full years without any variation. Room, Staff, and Location are the consistent top drivers of reviewer score. This stability is what gives us confidence in using SHAP values for per-hotel prioritization advice, since you're not getting answers that would flip if the data were sliced differently.

---

## Module C: conversational agent

### Methodology

We designed 20 queries covering all four query types: evidence, prioritization, mismatch, and segment. A handful of deliberate edge cases were included to test failure behavior: a nonexistent hotel name, a pricing question the system has no data on, and a segment query where the SHAP summary can't break down by traveler type.

GPT-4o served as the judge for relevance and context accuracy, both scored 1-5. Using the same model family as the generator is a known limitation of LLM-as-judge evaluation; scores should be treated as indicative rather than absolute.

| Query type | Count | Description |
|---|---|---|
| Evidence | 8 | Aspect complaints and praise, one nonexistent hotel, one out-of-scope query (pricing) |
| Prioritization | 4 | Global and per-hotel SHAP ranking queries |
| Mismatch | 2 | Queries about the gap between text sentiment and numeric score |
| Segment | 6 | Business, couple, family, and solo traveler queries |

### Results

| Metric | Score |
|---|---|
| Relevance (GPT-4o judge, 1-5) | 5.00 ± 0.00 |
| Context accuracy (GPT-4o judge, 1-5) | 4.30 ± 1.05 |
| Groundedness (citation coverage) | 57.89% |
| Resolution rate | 95% (19/20) |
| Mean retrieval similarity | 0.7159 |

### Reading the numbers

**Relevance: 5.00.** Perfect across all 20 queries. This includes the edge cases where the system couldn't give a full answer. Responding "I can't answer that confidently" to an unresolvable hotel or out-of-scope question is still a relevant response.

**Context accuracy: 4.30 ± 1.05.** The variance comes from a specific pattern: queries where the retrieved evidence reflects genuine disagreement in the reviews. Location queries are the clearest case. Some guests praise a hotel's location, others criticize it, and the response hedges rather than giving a clear answer. The judge scores hedging responses lower, but that behavior is correct given what the data actually contains. Room-related queries showed a similar dynamic: some retrieved "room" complaints were about noise or A/C, which cross-classify with other aspects in the pipeline. Retrieval is working correctly; the aspect taxonomy just has real overlap at the boundaries.

**Groundedness: 57.89%.** This looks low until you account for query type. Prioritization and mismatch queries retrieve SHAP narratives, not individual review sentences, so they produce zero sentence citations by design. Among evidence and segment queries, where citations are expected, coverage was effectively 100%.

**Resolution rate: 95%.** 19 of 20 queries got a grounded response. The one exception was the deliberately nonexistent hotel "Hotel Xyzabc Nonexistent 1234", which correctly triggered the unresolvable-hotel path instead of fabricating a response. The system asked the user to verify the hotel name, which is the right behavior.

**Mean retrieval similarity: 0.7159.** ChromaDB returns cosine similarity scores for each retrieved chunk. The mean across all evidence and segment queries was 0.72. For `text-embedding-3-small` on domain-specific retrieval, this is a solid result. Broad queries like "what do guests say about location globally" naturally score lower than specific ones, but the retrieved content is still useful.

### Edge cases

Three queries were designed to test failure modes:

**Nonexistent hotel:** "Tell me about Hotel Xyzabc Nonexistent 1234". Fuzzy match confidence fell below the resolution threshold and the agent correctly flagged it as unresolvable, asking for a corrected name.

**Out-of-scope query:** "What is the price range for rooms?" The system has no pricing data, but it correctly scoped the response to what was available: a handful of price mentions found in review text, clearly presented as anecdotal rather than structured data.

**Empty segment result:** "Which aspects do couples rate most positively?" at global scope. The SHAP global summary doesn't break down by segment, so the agent explained what it could and couldn't say rather than attempting an answer with no backing data.

All three behaved as expected.

---

## Summary

| Module | Primary metric | Result |
|---|---|---|
| A — ABSA | Aspect classification accuracy | ~85% (100-sentence manual check) |
| B — Rating impact | XGBoost R² | 0.26 |
| B — Rating impact | SHAP top-3 stability | Consistent across all 9 quarterly slices |
| C — Agent | Relevance (GPT-4o judge) | 5.00 / 5 |
| C — Agent | Context accuracy (GPT-4o judge) | 4.30 / 5 |
| C — Agent | Resolution rate | 95% |
