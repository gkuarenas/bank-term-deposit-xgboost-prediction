# Bank Term Subscription Prediction (XGBoost)

End-to-end machine learning project that predicts whether a client will subscribe to a **bank term deposit** (`y ∈ {yes,no}`) based on demographics, financial profile, and marketing campaign attributes. The project covers the full workflow: data prep → modeling → experiment tracking → modular pipelines → inference (API) → UI.

---

## Project Goals

- Build a production-style ML workflow (not just a notebook).
- Compare multiple models (RF / LightGBM / XGBoost) and select the best based on validation metrics.
- Track experiments and artifacts using **MLflow**.
- Keep code modular (separate data loading, validation, preprocessing, training, inference).
- Serve predictions via **API** (and optionally a simple UI).

---

## Dataset

Source: UCI Machine Learning Repository — **Bank Marketing (id=222)**  
- URL: `https://archive.ics.uci.edu/dataset/222/bank+marketing`
- Size: 45,211 rows
- Target: `y` (term deposit subscribed: `yes`/`no`)
- License: CC BY 4.0 (per UCI listing)
- Citation (APA):
  - Moro, S., Rita, P., & Cortez, P. (2014). Bank Marketing [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5K306

### Feature groups (summary)

**Bank client data**
- `age` (numeric)
- `job` (categorical)
- `marital` (categorical)
- `education` (categorical)
- `default` (binary)
- `balance` (numeric)
- `housing` (binary)
- `loan` (binary)

**Last contact of the current campaign**
- `contact` (categorical)
- `day` (numeric)
- `month` (categorical)
- `duration` (numeric)

**Other campaign attributes**
- `campaign` (numeric)
- `pdays` (numeric)
- `previous` (numeric)
- `poutcome` (categorical)

---

## What was done

### 1) Exploratory Data Analysis (EDA)
- Basic distribution checks
- Target balance review (subscription class imbalance)
- Quick sanity checks for missing/invalid values

### 2) Model comparison
Compared:
- Random Forest
- LightGBM
- XGBoost

Chosen model:
- **XGBoost** (best balance of metrics on validation set)

### 3) Train/Validation/Test split
- **75% / 15% / 10%** split (train / validation / test)

### 4) Hyperparameter optimization
- **Optuna** for tuning key XGBoost hyperparameters
- Focus on improving metrics for the positive class (`y = yes`)

### 5) Experiment tracking
- **MLflow** used to log:
  - parameters
  - metrics
  - model artifact
  - run metadata (timing, etc.)

### 6) Modular pipeline
Main orchestration via a pipeline script (e.g., `run_pipeline.py`), with modules for:
- data loading (e.g., `load_data.py`)
- validation (e.g., `validate_data.py`)
- preprocessing / feature engineering (e.g., `preprocessing.py`)
- training + evaluation
- inference pipeline

### 7) Feature preprocessing
- Deterministic binary encoding for binary columns
- One-hot encoding for categorical columns
- Schema consistency checks so training/serving transformations match

---

## Results (Positive class: `y = yes`)

- Precision: **0.5629**
- Recall: **0.5577**
- F1-score: **0.5603**
- Training time: **2.19s**
- Inference time: **0.02s**

Notes:
- Metrics reflect the practical “yes” class performance, which is often the harder class in this dataset due to imbalance.

---

## Quickstart

### 1) Create environment + install dependencies
Windows (PowerShell):
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```
### 2) Run Pipeline
```bash
python scripts/run_pipeline.py
```
or
```bash
mlflow ui
python scripts/run_pipeline.py --mlflow_uri http://127.0.0.1:5000 --run_name run_n
```
**Note**: replace run_n with your own name for logging

### 3) Run FastAPI app
```bash
fastapi dev src/app/main.py
```
Then once runnning, go to `http://127.0.0.1:8000/ui`
---

## Common Problems Encountered (and fixes)

### 1) Low XGBoost performance even after tuning

**What happened:**
- Hyperparameter optimization improved some runs but overall “yes” class performance remained modest.

### 2) File not found / directory issues (especially on Windows)

**What happened:**
- Path issues and working-directory confusion (scripts executed from different locations).

**Fixes:**
- Always run commands from the project root (where `src/`, `scripts/`, `requirements.txt` live).
- Use absolute or project-root-relative paths inside scripts.
- Confirm the pipeline creates required directories before saving artifacts (e.g., `artifacts/`, `models/`).

---

## Next Improvements (Roadmap)

- Add stricter data validation (types, ranges, allowed categories) beyond “all columns present”.
- Add unit tests for preprocessing and schema consistency.
- Improvement to the ML prediction model.
- Add threshold tuning + PR-AUC tracking for imbalanced classification.
- Add CI (GitHub Actions) to run lint + tests on every push.
- *Finish containerizing* training/inference end-to-end with Docker.
