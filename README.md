# Hotel Review Intelligence

BT5153 Applied Machine Learning in Business Analytics, Group 16

Takes 515k unstructured hotel reviews and makes them queryable. Ask the chatbot "what do solo travellers say about the rooms here?" and it pulls answers from real review sentences, with sources.

The pipeline runs in five stages: sentences are split from positive and negative review fields, classified into six aspects (cleanliness, staff, location, noise, food, room), and assigned sentiment. A linear regression and XGBoost model then fits those aspect scores to guest ratings, and SHAP shows which aspects drag ratings down most per hotel. The agent retrieves from two ChromaDB collections using HyDE query expansion and a LangGraph DAG, then cites the actual review sentences behind every claim.

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

## Pipeline stages

Stage 1 (`src/absa/preprocess.py`) splits each review's positive and negative text fields into individual sentences, tags each sentence with its source polarity and reviewer segment (Business, Couple, Family, Solo, Group derived from the Tags column), and writes `outputs/sentences.csv` (about 836k rows).

Stage 2 (`src/absa/aspect_extraction.py`) runs LDA topic modeling on the sentence corpus to build a vocabulary dictionary mapping words to six aspects: cleanliness, staff, location, noise, food, and room.

Stage 3 (`src/absa/sentiment_assignment.py`) keyword-matches each sentence to an aspect and assigns a sentiment score (+1 positive, -1 negative, 0 not mentioned) based on which review field the sentence came from. It writes `outputs/aspect_sentences.csv` with sentence-level labels and `outputs/review_features.csv` with one row per review.

Stage 4 (`src/rating_impact/model.py`, `evaluate.py`) trains a linear regression and an XGBoost model on the aspect feature matrix to predict reviewer scores. SHAP values are computed globally and per hotel. Results go to `outputs/shap_summary.json` and `outputs/impact_report.csv`.

Stage 5 (`src/agent/ingest.py`) embeds all ~836k sentences into ChromaDB `evidence_store` using `text-embedding-3-small`, and indexes SHAP narratives into `summary_store`. You run this once after the full pipeline completes. The embedding costs around $0.40.

## Agent

The chatbot uses a LangGraph DAG with eight nodes. You can ask four kinds of questions.

For evidence questions ("What do guests complain about regarding noise?"), the agent generates a hypothetical ideal answer, embeds it, and queries `evidence_store` for the closest matching sentences. This is HyDE — embedding a hypothetical answer rather than the raw query produces better semantic matches against real review text.

For prioritization questions ("Which aspect hurts this hotel's ratings the most?"), the agent skips embedding entirely and fetches the SHAP ranking directly from `summary_store`.

Segment questions ("What do business travellers say about the rooms?") add a `reviewer_segment` metadata filter before retrieval so results are scoped to that guest type.

Follow-up questions reuse the hotel and aspect context from the prior turn, so you don't have to repeat yourself across a conversation.

The Gradio UI has a hotel selector dropdown. Selecting a hotel scopes all retrieval to that hotel's reviews. "All Hotels" runs against the full dataset.

Every answer cites the retrieved sentences with hotel, aspect, sentiment, and the raw quote.

## Project structure

```
src/absa/             aspect extraction and sentiment assignment (Stages 1-3)
src/rating_impact/    regression modeling and SHAP analysis (Stage 4)
src/agent/            LangGraph RAG agent with ChromaDB (Stages 5-6)
  nodes/              one file per LangGraph node
src/ui/               Gradio chatbot interface
scripts/              run_pipeline.py orchestrates all stages
outputs/              generated pipeline artifacts (gitignored for large files)
data/                 raw data (not committed, add data.csv here)
notebooks/            exploratory analysis
docs/                 internal design documents
```

See `architecture.md` for the full pipeline diagram, LangGraph node descriptions, and ChromaDB schema.

## Dataset

515k European hotel reviews across 1,492 hotels. Not committed (228MB, over GitHub's limit).

**To set up locally:**
1. Download `data.csv` from the shared NUS OneDrive link in `docs/DETAILS.md`
2. Place it at `data/data.csv`
3. Create venv with the requirements.txt (i'll add a local_startup batch file to make running easier for evaluation later on)
4. Additionally, in the OneDrive -> outputs are classified by stage, copy all or relevant outputs and place them in the outputs folder - path issues have been resolved and there are load_X() functions to make loading df's easier.

Source: [Kaggle, 515K Hotel Reviews Data in Europe](https://www.kaggle.com/datasets/jiashenliu/515k-hotel-reviews-data-in-europe)

## EDA notebook

Exploratory analysis is in `notebooks/EDA_and_Stage1_Text_Preparation.ipynb`. It does not write any output files.

```bash
# place data.csv in notebooks/ first
source .venv/Scripts/activate   # Windows Git Bash
jupyter notebook notebooks/
```

Select the **BT5153 (venv)** (or your local venv name) as kernel and run all cells.

## Course

BT5153 Applied Machine Learning in Business Analytics
National University of Singapore, Group 16
