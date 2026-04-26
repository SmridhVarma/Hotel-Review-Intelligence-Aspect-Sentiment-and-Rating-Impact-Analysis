import os
import sys
import json
import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import shap
import xgboost as xgb

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

ASPECTS = ["cleanliness", "staff", "location", "noise", "food", "room"]
_MIN_HOTEL_REVIEWS = 100


def load_review_features() -> pd.DataFrame:
    """Load the Stage 3 per-review aspect feature matrix."""
    return pd.read_csv(REVIEW_FEATURES, low_memory=False)


# 485,646 rows — one row per review with aspect sentiment scores and target.
# Columns: review_id, hotel_name, reviewer_score (target),
#          cleanliness, staff, location, noise, food, room (+1 / -1 / 0)
# Available after Stage 3 (sentiment_assignment.py) has been run.
df = load_review_features() if os.path.isfile(REVIEW_FEATURES) else None


def train_models(df: pd.DataFrame):
    """
    Train LinearRegression and XGBRegressor on aspect features.
    Returns (linear, xgb_model, X_train, X_test, y_train, y_test).
    """
    X = df[ASPECTS].values.astype(float)
    y = df["reviewer_score"].values.astype(float)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    linear = LinearRegression()
    linear.fit(X_train, y_train)

    xgb_model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=1.0,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    xgb_model.fit(X_train, y_train)

    return linear, xgb_model, X_train, X_test, y_train, y_test


def compute_shap_values(model, X: np.ndarray, model_type: str = "xgb") -> np.ndarray:
    """Compute SHAP values for all rows; use TreeExplainer for XGB, LinearExplainer otherwise."""
    if model_type == "xgb":
        explainer = shap.TreeExplainer(model)
        return explainer.shap_values(X)
    explainer = shap.LinearExplainer(model, X)
    return explainer.shap_values(X)


def build_shap_summary(df: pd.DataFrame, shap_values: np.ndarray) -> list[dict]:
    """
    Aggregate per-review SHAP values to per-hotel means plus a weighted global entry.
    df must be reset-indexed so row positions align with shap_values.
    """
    df = df.reset_index(drop=True)
    results = []

    for hotel_name, group in df.groupby("hotel_name"):
        idx = group.index.tolist()
        hotel_shap = shap_values[idx]
        review_count = len(idx)
        entry = {
            "hotel_name": hotel_name,
            "aspect_impacts": {
                asp: float(np.mean(hotel_shap[:, i]))
                for i, asp in enumerate(ASPECTS)
            },
            "review_count": review_count,
        }
        if review_count < _MIN_HOTEL_REVIEWS:
            entry["insufficient_data"] = True
        results.append(entry)

    total = sum(r["review_count"] for r in results)
    global_entry = {
        "hotel_name": "__global__",
        "aspect_impacts": {
            asp: float(
                sum(r["aspect_impacts"][asp] * r["review_count"] for r in results) / total
            )
            for asp in ASPECTS
        },
        "review_count": total,
    }
    return [global_entry] + results


def save_outputs(
    linear: LinearRegression,
    xgb_model: xgb.XGBRegressor,
    shap_summary: list[dict],
) -> None:
    """Serialize models and write shap_summary.json + impact_report.csv."""
    with open(LINEAR_MODEL, "wb") as f:
        pickle.dump(linear, f)
    with open(XGB_MODEL, "wb") as f:
        pickle.dump(xgb_model, f)

    # shap_summary.json — strip internal review_count before writing
    summary_out = []
    for entry in shap_summary:
        out = {
            "hotel_name": entry["hotel_name"],
            "aspect_impacts": entry["aspect_impacts"],
        }
        if entry.get("insufficient_data"):
            out["insufficient_data"] = True
        summary_out.append(out)
    with open(SHAP_SUMMARY, "w") as f:
        json.dump(summary_out, f, indent=2)

    # impact_report.csv — long format, ranked by |shap_value| descending
    rows = []
    for entry in shap_summary:
        hotel = entry["hotel_name"]
        ranked = sorted(
            entry["aspect_impacts"].items(), key=lambda x: abs(x[1]), reverse=True
        )
        for rank, (asp, val) in enumerate(ranked, 1):
            rows.append({
                "hotel_name": hotel,
                "aspect": asp,
                "shap_value": round(val, 6),
                "rank": rank,
            })
    pd.DataFrame(rows).to_csv(IMPACT_REPORT, index=False)


def run() -> None:
    print("[Stage 4 / model] Loading review features...")
    data = load_review_features()
    data = data.reset_index(drop=True)
    print(f"  {len(data):,} reviews loaded")

    print("[Stage 4 / model] Training LinearRegression and XGBRegressor...")
    linear, xgb_model, *_ = train_models(data)
    print("  Training complete")

    print("[Stage 4 / model] Computing SHAP values (XGBoost, full dataset)...")
    X_all = data[ASPECTS].values.astype(float)
    shap_values = compute_shap_values(xgb_model, X_all, model_type="xgb")
    print(f"  SHAP matrix: {shap_values.shape}")

    print("[Stage 4 / model] Aggregating SHAP by hotel...")
    summary = build_shap_summary(data, shap_values)
    n_hotels = len(summary) - 1  # exclude __global__
    print(f"  {n_hotels} hotels aggregated")

    print("[Stage 4 / model] Saving outputs...")
    save_outputs(linear, xgb_model, summary)
    print(f"  {LINEAR_MODEL}")
    print(f"  {XGB_MODEL}")
    print(f"  {SHAP_SUMMARY}")
    print(f"  {IMPACT_REPORT}")
    print("[Stage 4 / model] Done.")


if __name__ == "__main__":
    run()
