import os
import sys
import pickle

import pandas as pd

_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from paths import (
    REVIEW_FEATURES,
    LINEAR_MODEL,
    XGB_MODEL,
    EVALUATION_REPORT,
)


def load_review_features() -> pd.DataFrame:
    """Load the Stage 3 per-review aspect feature matrix."""
    return pd.read_csv(REVIEW_FEATURES, low_memory=False)


def load_models():
    """Load trained linear and XGBoost models from model_artifacts/."""
    with open(LINEAR_MODEL, "rb") as f:
        linear = pickle.load(f)
    with open(XGB_MODEL, "rb") as f:
        xgb = pickle.load(f)
    return linear, xgb


# 515,738 rows — same feature matrix used for training; split here for evaluation.
# Columns: review_id, hotel_name, reviewer_score (target),
#          cleanliness, staff, location, noise, food, room (+1 / -1 / 0)
# Available after Stage 3 (sentiment_assignment.py) has been run.
df = load_review_features() if os.path.isfile(REVIEW_FEATURES) else None

# Trained LinearRegression and XGBRegressor from outputs/model_artifacts/
# Available after Stage 4 (model.py) has been run.
linear_model, xgb_model = load_models() if os.path.isfile(LINEAR_MODEL) else (None, None)


# evaluate.py — Model Evaluation
#
# Purpose:
#   Evaluates trained models on a held-out test set and checks SHAP
#   stability across time slices.
#
# Input:
#   outputs/model_artifacts/linear_model.pkl
#   outputs/model_artifacts/xgb_model.pkl
#   outputs/review_features.csv
#     (same schema as model.py input)
#
# Output:
#   outputs/evaluation_report.json
#     Schema:
#     {
#       "linear_regression": { "rmse": float, "mae": float, "r2": float },
#       "xgboost":           { "rmse": float, "mae": float, "r2": float },
#       "shap_stability": {
#         "top_3_consistent_across_slices": bool,
#         "slice_rankings": [ { "period": str, "top_aspects": [str] } ]
#       }
#     }
