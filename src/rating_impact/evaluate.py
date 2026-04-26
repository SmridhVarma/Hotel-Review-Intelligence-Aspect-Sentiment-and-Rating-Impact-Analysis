import os
import sys
import json
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import shap

_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from paths import (
    REVIEW_FEATURES,
    SENTENCES,
    LINEAR_MODEL,
    XGB_MODEL,
    EVALUATION_REPORT,
)

ASPECTS = ["cleanliness", "staff", "location", "noise", "food", "room"]


def load_review_features() -> pd.DataFrame:
    """Load the Stage 3 per-review aspect feature matrix."""
    return pd.read_csv(REVIEW_FEATURES, low_memory=False)


def load_models():
    """Load trained linear and XGBoost models from model_artifacts/."""
    with open(LINEAR_MODEL, "rb") as f:
        linear = pickle.load(f)
    with open(XGB_MODEL, "rb") as f:
        xgb_model = pickle.load(f)
    return linear, xgb_model


# 485,646 rows — same feature matrix used for training; split here for evaluation.
# Columns: review_id, hotel_name, reviewer_score (target),
#          cleanliness, staff, location, noise, food, room (+1 / -1 / 0)
# Available after Stage 3 (sentiment_assignment.py) has been run.
df = load_review_features() if os.path.isfile(REVIEW_FEATURES) else None

# Trained LinearRegression and XGBRegressor from outputs/model_artifacts/
# Available after Stage 4 (model.py) has been run.
linear_model, xgb_model = load_models() if os.path.isfile(LINEAR_MODEL) else (None, None)


def _evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    y_pred = model.predict(X_test)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "r2": float(r2_score(y_test, y_pred)),
    }


def _shap_stability(model, df_dated: pd.DataFrame) -> tuple[bool, list[dict]]:
    """
    Compute per-quarter SHAP top-3 aspects and check consistency across slices.
    Ranks by absolute mean SHAP (importance magnitude, not direction).
    """
    df_dated = df_dated.copy()
    df_dated["period"] = (
        pd.to_datetime(df_dated["review_date"])
        .dt.to_period("Q")
        .astype(str)
    )
    explainer = shap.TreeExplainer(model)
    slice_rankings = []

    for period, group in sorted(df_dated.groupby("period")):
        X_slice = group[ASPECTS].values.astype(float)
        sv = explainer.shap_values(X_slice)
        mean_abs = np.abs(sv).mean(axis=0)
        top3 = [ASPECTS[i] for i in np.argsort(mean_abs)[::-1][:3]]
        slice_rankings.append({"period": period, "top_aspects": top3})

    if len(slice_rankings) >= 2:
        first = set(slice_rankings[0]["top_aspects"])
        consistent = all(set(s["top_aspects"]) == first for s in slice_rankings)
    else:
        consistent = True  # trivially consistent with a single slice

    return consistent, slice_rankings


def run() -> None:
    print("[Stage 4 / evaluate] Loading data and models...")
    data = load_review_features()
    linear, xgb_mdl = load_models()

    X = data[ASPECTS].values.astype(float)
    y = data["reviewer_score"].values.astype(float)
    # Reproduce the same 80/20 split used in model.py (random_state=42)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("[Stage 4 / evaluate] Evaluating on held-out test set...")
    results = {
        "linear_regression": _evaluate_model(linear, X_test, y_test),
        "xgboost": _evaluate_model(xgb_mdl, X_test, y_test),
    }
    for name, m in results.items():
        print(f"  {name}: RMSE={m['rmse']:.4f}  MAE={m['mae']:.4f}  R²={m['r2']:.4f}")

    print("[Stage 4 / evaluate] Checking SHAP stability across time slices...")
    stability: dict = {"top_3_consistent_across_slices": None, "slice_rankings": []}

    if os.path.isfile(SENTENCES):
        # Read only the columns needed for date enrichment — avoids loading 155 MB of text
        dates = (
            pd.read_csv(SENTENCES, usecols=["review_id", "review_date"], low_memory=False)
            .drop_duplicates(subset="review_id")[["review_id", "review_date"]]
        )
        df_dated = data.merge(dates, on="review_id", how="left").dropna(subset=["review_date"])
        consistent, slices = _shap_stability(xgb_mdl, df_dated)
        stability = {
            "top_3_consistent_across_slices": consistent,
            "slice_rankings": slices,
        }
        print(f"  {len(slices)} time slices; consistent top-3: {consistent}")
    else:
        print(f"  {SENTENCES} not found — skipping time-slice stability check")

    report = {**results, "shap_stability": stability}
    with open(EVALUATION_REPORT, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Saved: {EVALUATION_REPORT}")
    print("[Stage 4 / evaluate] Done.")


if __name__ == "__main__":
    run()
