# Hotel Review Intelligence

**BT5153 Applied Machine Learning in Business Analytics — Group 16**

A machine learning system that transforms unstructured hotel reviews into structured aspect-level insights, exposed through a conversational AI agent for hotel owners.

## What It Does

- **Aspect Extraction** — identifies 6 service aspects (Cleanliness, Staff, Location, Noise, Food, Room) from 515k hotel reviews using LDA topic modeling
- **Rating Impact Analysis** — quantifies which aspects most strongly drive guest ratings using regression + SHAP values
- **Conversational Agent** — a LangGraph RAG chatbot grounded in real review evidence, queryable by hotel and reviewer segment

## Setup

```bash
# 1. Clone the repo
git clone <repo-url>
cd hotel-review-intelligence

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# 5. Add data
# Place data.csv in the data/ directory
```

## Running the Pipeline

```bash
# Full pipeline (~1-2 hours for 515k reviews including embedding)
python scripts/run_pipeline.py

# Launch the chatbot UI (after pipeline completes)
python src/ui/app.py
# Open http://localhost:7860
```

## Project Structure

```
src/absa/             Aspect extraction and sentiment assignment (Stages 1–3)
src/rating_impact/    Regression modeling and SHAP analysis (Stage 4)
src/agent/            LangGraph RAG agent with ChromaDB (Stages 5–6)
  nodes/              One file per LangGraph node
src/ui/               Gradio chatbot interface
scripts/              run_pipeline.py orchestrates all stages
outputs/              Generated pipeline artifacts
data/                 Raw data (not committed — add data.csv here)
notebooks/            Exploratory analysis
```

## Dataset

515,000 European hotel reviews across 1,492 hotels. The file is **not committed** (228MB — exceeds GitHub's limit).

**To set up locally:**
1. Download `data.csv` from the shared NUS OneDrive link (see group chat)
2. Place it at `data/data.csv`

Source: [Kaggle — 515K Hotel Reviews Data in Europe](https://www.kaggle.com/datasets/jiashenliu/515k-hotel-reviews-data-in-europe)

## EDA Notebook

Exploratory analysis and data cleaning inspection is in `notebooks/EDA_and_Stage1_Text_Preparation.ipynb`.

```bash
# 1. Place data.csv in the notebooks/ folder
# 2. Activate the venv and launch Jupyter
source .venv/Scripts/activate        # Windows Git Bash
jupyter notebook notebooks/
```

Open `EDA_and_Stage1_Text_Preparation.ipynb`, select the **BT5153 (venv)** kernel, and run all cells.
The notebook is read-only exploration — it does not write any output files.

## Course

BT5153 Applied Machine Learning in Business Analytics  
National University of Singapore — Group 16
