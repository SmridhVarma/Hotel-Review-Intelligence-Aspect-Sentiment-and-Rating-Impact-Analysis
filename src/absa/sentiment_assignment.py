# sentiment_assignment.py — Stage 3: Aspect-Level Sentiment Assignment
#
# Purpose:
#   Keyword-matches each sentence to an aspect, then assigns sentiment
#   from source polarity. Produces both a sentence-level labeled dataset
#   and a review-level feature matrix for downstream modeling.
#
# Input:
#   outputs/sentences.csv
#     review_id        (int)  : review identifier
#     hotel_name       (str)  : hotel identifier
#     sentence         (str)  : sentence text
#     source_polarity  (str)  : "positive" | "negative"
#     reviewer_segment (str)  : guest segment
#     reviewer_score   (float): numeric rating
#
#   outputs/aspect_dictionary.json
#     Schema: { aspect_name: [keyword, ...] }
#
# Output:
#   outputs/aspect_sentences.csv
#     review_id        (int)  : review identifier
#     hotel_name       (str)  : hotel identifier
#     sentence         (str)  : sentence text
#     aspect           (str)  : matched aspect | null if no match
#     sentiment        (int)  : +1 (positive) | -1 (negative) | 0 (no match)
#     reviewer_segment (str)  : guest segment
#     reviewer_score   (float): numeric rating
#
#   outputs/review_features.csv
#     review_id    (int)  : review identifier
#     hotel_name   (str)  : hotel identifier
#     reviewer_score (float): numeric rating — prediction target for model.py
#     cleanliness  (int)  : +1 | -1 | 0
#     staff        (int)  : +1 | -1 | 0
#     location     (int)  : +1 | -1 | 0
#     noise        (int)  : +1 | -1 | 0
#     food         (int)  : +1 | -1 | 0
#     room         (int)  : +1 | -1 | 0
