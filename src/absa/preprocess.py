# preprocess.py — Stage 1: Text Preparation
#
# Purpose:
#   Combines positive and negative review fields into a single text per review,
#   sentence-tokenizes each review, and tags each sentence with source polarity
#   and reviewer segment.
#
# Input:
#   data/data.csv
#     Hotel_Name       (str) : hotel identifier
#     Positive_Review  (str) : guest's positive comments
#     Negative_Review  (str) : guest's negative comments
#     Reviewer_Score   (float): numeric rating 0–10
#     Tags             (str) : trip type tags e.g. "[' Leisure trip ', ' Couple ']"
#     Review_Date      (str) : date of review
#
# Output:
#   outputs/sentences.csv
#     review_id        (int)  : row index from source data
#     hotel_name       (str)  : hotel identifier
#     sentence         (str)  : individual sentence from review
#     source_polarity  (str)  : "positive" | "negative"
#     reviewer_score   (float): original numeric rating
#     reviewer_segment (str)  : Business | Couple | Family | Solo | Group | Unknown
#     review_date      (str)  : original review date
