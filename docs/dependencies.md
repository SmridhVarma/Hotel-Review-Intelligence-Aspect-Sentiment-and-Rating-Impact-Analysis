# Pipeline dependencies and ownership

## Stage map

| Stage | Script | Inputs | Outputs | Owner |
|---|---|---|---|---|
| 1 | `src/absa/preprocess.py` | `data/data.csv` | `outputs/sentences.csv` `outputs/clean_reviews_stage1.csv` | Module A |
| 2 | `src/absa/aspect_extraction.py` | `outputs/sentences.csv` | `outputs/aspect_dictionary.json` | Module A |
| 3 | `src/absa/sentiment_assignment.py` | `outputs/sentences.csv` `outputs/aspect_dictionary.json` | `outputs/aspect_sentences.csv` `outputs/review_features.csv` | Module A |
| 4 | `src/rating_impact/model.py` | `outputs/review_features.csv` | `outputs/model_artifacts/linear_model.pkl` `outputs/model_artifacts/xgb_model.pkl` `outputs/shap_summary.json` `outputs/impact_report.csv` | Module B |
| 4 | `src/rating_impact/evaluate.py` | `outputs/review_features.csv` `outputs/model_artifacts/*.pkl` | `outputs/evaluation_report.json` | Module B |
| 5 | `src/agent/ingest.py` | `outputs/aspect_sentences.csv` `outputs/shap_summary.json` | `chromadb/evidence_store` `chromadb/summary_store` | Module C |
| 6 | `src/agent/graph.py` + `nodes/` | `chromadb/` (via ChromaDB client) | agent responses + citations | Module C |
| UI | `src/ui/app.py` | `src/agent/graph.py` (imported) | Gradio interface at `localhost:7860` | Module D |

## Cross-module file contracts

| File | Produced by | Consumed by | Gitignored | Notes |
|---|---|---|---|---|
| `data/data.csv` | — (external) | Stage 1 | yes | 228MB, download from OneDrive |
| `outputs/sentences.csv` | Stage 1 | Stage 2, Stage 3 | yes | ~155MB |
| `outputs/clean_reviews_stage1.csv` | Stage 1 | EDA notebook only | yes | ~302MB |
| `outputs/aspect_dictionary.json` | Stage 2 | Stage 3 | no | committed |
| `outputs/aspect_sentences.csv` | Stage 3 | Stage 5 | yes | ~129MB, share via OneDrive |
| `outputs/review_features.csv` | Stage 3 | Stage 4 | yes | ~30MB, share via OneDrive |
| `outputs/shap_summary.json` | Stage 4 | Stage 5 | no | committed once produced |
| `outputs/impact_report.csv` | Stage 4 | — (report only) | no | committed once produced |
| `outputs/evaluation_report.json` | Stage 4 | — (report only) | no | committed once produced |
| `chromadb/` | Stage 5 | Stage 6 / UI | yes | rebuild locally via ingest.py |

## Critical path to a running agent

```
data.csv
  └─ Stage 1 ──► sentences.csv
                   └─ Stage 2 ──► aspect_dictionary.json
                   └─ Stage 3 ──► aspect_sentences.csv ──► Stage 5 ──► chromadb/ ──► Stage 6
                                  review_features.csv
                                    └─ Stage 4 ──► shap_summary.json ──► Stage 5
```

Stage 5 (ingest) is the last blocking dependency before the agent runs. It needs both `aspect_sentences.csv` (Module A) and `shap_summary.json` (Module B).

## Current status

| Stage | Status |
|---|---|
| Stage 1 | complete — outputs in OneDrive |
| Stage 2 | complete — `aspect_dictionary.json` committed |
| Stage 3 | complete — outputs in OneDrive |
| Stage 4 | stub — `model.py` and `evaluate.py` not yet implemented |
| Stage 5 | stub — `ingest.py` not yet implemented |
| Stage 6 | stub — all node files not yet implemented |
| UI | stub — `app.py` not yet implemented |
