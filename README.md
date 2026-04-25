# Hotel Review Intelligence

BT5153 Applied Machine Learning in Business Analytics, Group 16

Takes 515k unstructured hotel reviews and makes them queryable. Ask the chatbot "what do solo travellers say about the rooms here?" and it pulls answers from real review sentences, with sources.

The pipeline runs in four stages before the agent: sentences are split from positive and negative review fields, classified into six aspects (cleanliness, staff, location, noise, food, room), and assigned sentiment. A regression and XGBoost model then fits those aspect scores to guest ratings, and SHAP shows which aspects matter most per hotel. The agent retrieves from ChromaDB using HyDE and a LangGraph DAG.

## Setup

```bash
git clone <repo-url>
cd hotel-review-intelligence

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env
# add your OPENAI_API_KEY to .env

# place data.csv in the data/ directory
```

## Running the pipeline

```bash
# full pipeline, ~1-2 hours for 515k reviews including embedding
python scripts/run_pipeline.py

# launch the chatbot after the pipeline finishes
python src/ui/app.py
# open http://localhost:7860
```

## Project structure

```
src/absa/             aspect extraction and sentiment assignment (Stages 1-3)
src/rating_impact/    regression modeling and SHAP analysis (Stage 4)
src/agent/            LangGraph RAG agent with ChromaDB (Stages 5-6)
  nodes/              one file per LangGraph node
src/ui/               Gradio chatbot interface
scripts/              run_pipeline.py orchestrates all stages
outputs/              generated pipeline artifacts
data/                 raw data (not committed, add data.csv here)
notebooks/            exploratory analysis
```

## Dataset

515k European hotel reviews across 1,492 hotels. Not committed (228MB, over GitHub's limit).

**To set up locally:**
1. Download `data.csv` from the shared NUS OneDrive link in docs/DETAILS.md
2. Place it at `data/data.csv`

Source: [Kaggle, 515K Hotel Reviews Data in Europe](https://www.kaggle.com/datasets/jiashenliu/515k-hotel-reviews-data-in-europe)

## EDA notebook

Exploratory analysis is in `notebooks/EDA_and_Stage1_Text_Preparation.ipynb`. It does not write any output files.

```bash
# place data.csv in notebooks/ first
source .venv/Scripts/activate   # Windows Git Bash
jupyter notebook notebooks/
```

Select the **BT5153 (venv)** kernel and run all cells.

## Course

BT5153 Applied Machine Learning in Business Analytics
National University of Singapore, Group 16
