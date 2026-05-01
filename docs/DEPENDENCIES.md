# Pipeline dependencies and ownership
Refer this for assigned tasks and downstream files/outputs needed!

## Stage map

| Stage | Script | Inputs | Outputs | Owner |
|---|---|---|---|---|
| 1 | `src/absa/preprocess.py` | `data/data.xlsx` | `outputs/sentences.csv` `outputs/clean_reviews_stage1.csv` | Module A |
| 2 | `src/absa/aspect_extraction.py` | `outputs/sentences.csv` | `outputs/aspect_dictionary.json` | Module A |
| 3 | `src/absa/sentiment_assignment.py` | `outputs/sentences.csv` `outputs/aspect_dictionary.json` | `outputs/aspect_sentences.csv` `outputs/review_features.csv` | Module A |
| 4 | `src/rating_impact/model.py` | `outputs/review_features.csv` | `outputs/model_artifacts/linear_model.pkl` `outputs/model_artifacts/xgb_model.pkl` `outputs/shap_summary.json` `outputs/impact_report.csv` | Module B |
| 4 | `src/rating_impact/evaluate.py` | `outputs/review_features.csv` `outputs/model_artifacts/*.pkl` | `outputs/evaluation_report.json` | Module B |
| 5 | `src/agent/ingest.py` | `outputs/aspect_sentences.csv` `outputs/shap_summary.json` | `chromadb/evidence_store` `chromadb/summary_store` `outputs/hotel_names.json` | Module C |
| 6 | `src/agent/graph.py` + `nodes/` | `chromadb/` (via ChromaDB client) | agent responses + citations | Module C |
| UI | `src/ui/app.py` | `src/agent/graph.py` (imported) | Gradio interface at `localhost:7860` | Module D |

## Cross-module file contracts

| File | Produced by | Consumed by | Gitignored | Notes |
|---|---|---|---|---|
| `data/data.xlsx` | — (external) | Stage 1 | yes | 228MB, download from OneDrive |
| `outputs/sentences.csv` | Stage 1 | Stage 2, Stage 3 | yes | ~155MB, share via OneDrive |
| `outputs/clean_reviews_stage1.csv` | Stage 1 | EDA notebook only | yes | ~302MB, share via OneDrive |
| `outputs/aspect_dictionary.json` | Stage 2 | Stage 3 | no | committed |
| `outputs/aspect_sentences.csv` | Stage 3 | Stage 5 | yes | ~129MB, share via OneDrive |
| `outputs/review_features.csv` | Stage 3 | Stage 4 | yes | ~30MB, share via OneDrive |
| `outputs/shap_summary.json` | Stage 4 | Stage 5 | no | committed |
| `outputs/impact_report.csv` | Stage 4 | — (report only) | no | committed |
| `outputs/evaluation_report.json` | Stage 4 | — (report only) | no | committed |
| `outputs/hotel_names.json` | Stage 5 | Agent query_classifier | no | committed |
| `chromadb/` | Stage 5 | Stage 6 / UI | yes | pre-built on OneDrive; rebuilt by `local_startup.bat` if missing |

## Critical path to a running agent

```
data.xlsx
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
| Stage 4 | complete — models, SHAP, and evaluation report committed |
| Stage 5 | complete — `ingest.py` implemented; pre-built ChromaDB in OneDrive |
| Stage 6 | complete — all 8 LangGraph nodes implemented |
| UI | complete — `src/ui/app.py` Gradio chatbot, launched via `local_startup.bat` |
