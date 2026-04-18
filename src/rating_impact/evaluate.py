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
