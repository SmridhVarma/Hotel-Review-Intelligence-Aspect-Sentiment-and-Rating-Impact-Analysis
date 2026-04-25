import os
import sys

import pandas as pd

_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from paths import (
    REVIEW_FEATURES,
    ARTIFACTS_DIR,
    LINEAR_MODEL,
    XGB_MODEL,
    SHAP_SUMMARY,
    IMPACT_REPORT,
)

os.makedirs(ARTIFACTS_DIR, exist_ok=True)


def load_review_features() -> pd.DataFrame:
    """Load the Stage 3 per-review aspect feature matrix."""
    return pd.read_csv(REVIEW_FEATURES, low_memory=False)


# 515,738 rows — one row per review with aspect sentiment scores and target.
# Columns: review_id, hotel_name, reviewer_score (target),
#          cleanliness, staff, location, noise, food, room (+1 / -1 / 0)
# Available after Stage 3 (sentiment_assignment.py) has been run.
df = load_review_features() if os.path.isfile(REVIEW_FEATURES) else None


# model.py — Stage 4: Rating Impact Modeling
#
# Purpose:
#   Trains regression models on the aspect feature matrix to predict
#   reviewer_score. Computes SHAP values to rank aspect impact globally
#   and per hotel.
#
# Input:
#   outputs/review_features.csv
#     review_id      (int)  : review identifier
#     hotel_name     (str)  : hotel identifier
#     reviewer_score (float): prediction target
#     cleanliness    (int)  : aspect sentiment feature
#     staff          (int)  : aspect sentiment feature
#     location       (int)  : aspect sentiment feature
#     noise          (int)  : aspect sentiment feature
#     food           (int)  : aspect sentiment feature
#     room           (int)  : aspect sentiment feature
#
# Output:
#   outputs/model_artifacts/linear_model.pkl   : serialized LinearRegression
#   outputs/model_artifacts/xgb_model.pkl      : serialized XGBRegressor
#
#   outputs/shap_summary.json
#     Schema: list of objects, one per hotel + one "__global__" entry
#     [
#       {
#         "hotel_name": "__global__",
#         "aspect_impacts": {
#           "cleanliness": -0.42, "staff": 0.31, "location": 0.28,
#           "noise": -0.19, "food": 0.11, "room": -0.08
#         }
#       },
#       { "hotel_name": "Hotel Arena", "aspect_impacts": { ... } },
#       ...
#     ]
#     Values: mean SHAP values with sign (negative = drags rating down)
#
#   outputs/impact_report.csv
#     hotel_name  (str)  : hotel or "__global__"
#     aspect      (str)  : aspect name
#     shap_value  (float): mean SHAP value
#     rank        (int)  : 1 = highest impact
