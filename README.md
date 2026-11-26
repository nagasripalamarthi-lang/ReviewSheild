# 🛡️ ReviewShield – Fake Review Detection System

ReviewShield is a full-stack Python, NLP, and Machine Learning web application built with **Streamlit**, **Scikit-learn**, **Pandas**, **Plotly**, **Matplotlib**, and **SQLite**. It analyzes product reviews to flag potentially suspicious, spam, duplicated, or template-generated entries while explaining *why* each prediction was made.

> **Important Ethics & Data Disclaimer:**
> ReviewShield outputs **statistical model predictions** (`Potentially Genuine` vs. `Potentially Suspicious`), **not verified ground-truth proof** of fake reviews or fraud. The bundled dataset (`data/sample_reviews.csv`) is a **synthetic & curated educational benchmark** created to demonstrate NLP/ML fake-review detection patterns and does **not** represent real e-commerce customer records.

---

## ✨ Core Features

1. **🔍 Single Review Analysis**
   - Classify any individual review as **Potentially Genuine** or **Potentially Suspicious**.
   - Displays a **confidence score**, **suspicion probability gauge**, **human-readable linguistic signal cards** (promotional phrases, excessive punctuation, ALL-CAPS ratio, URL/contact detection, vocabulary diversity, concrete product-detail cues), and **top linear model feature drivers**.
   - Optional automatic logging to SQLite history.

2. **📂 Bulk CSV Detection**
   - Upload any CSV containing reviews (or load the built-in `data/bulk_test_reviews.csv` 30-review test batch in one click).
   - Auto-detects common review column names (`review_text`, `review`, `text`, `comment`, etc.) with manual override.
   - Separates **Potentially Suspicious** and **Potentially Genuine** reviews into dedicated tabs with confidence scores and explanation summaries.
   - Export full analyzed results or suspicious-only rows as CSV, and optionally save batches to SQLite.

3. **🧬 Duplicate & Pattern Detection**
   - Detects **exact normalized duplicates** across reviews.
   - Identifies **near-duplicate reviews** using TF-IDF cosine similarity with a configurable threshold slider and groups them into similarity clusters.
   - Extracts **repeated multi-word phrases (3–4 n-grams)** across reviews to uncover copy-paste spam templates.

4. **📊 Review Insights Dashboard**
   - Tracks total reviews analyzed, suspicious vs. genuine counts, suspicious review percentage, and average model confidence.
   - Interactive Plotly visualizations (classification donut chart, suspicion score histogram, word-count box plot, and word count vs. suspicion scatter plot).
   - Persistent SQLite storage (`data/reviewshield.db`) with one-click sample batch population and history management.

5. **🤖 ML Pipeline & Model Card**
   - Transparently documents preprocessing, leak-free train/test splitting, hybrid TF-IDF + 16-feature stylometric engineering, candidate model benchmarks (**Logistic Regression**, **Calibrated Linear SVM**, **Random Forest**, and **Complement Naive Bayes**), and a Matplotlib **confusion matrix**.

---

## 🧠 How the Machine Learning Pipeline Works

1. **Data Validation & Leak-Free Splitting**
   - `src/ml_model.py` validates missing/empty rows and performs a stratified **80% / 20%** train/test split *before* fitting any vectorizers or scalers.
2. **NLP Preprocessing & Stylometric Engineering (`src/nlp_preprocessing.py`)**
   - Cleans HTML tags, normalizes URLs/contact details to `urltoken`, and extracts 16 interpretable stylometric features (character/word counts, exclamation density, ALL-CAPS ratio, type-token vocabulary diversity, promotional/spam phrase hits, specific product-detail cues, first-person pronoun ratio, and word repetition).
3. **Hybrid Feature Union & Model Selection**
   - Combines **Word TF-IDF (1–2 n-grams)**, **Character TF-IDF (3–5 `char_wb` n-grams)**, and **StandardScaler-normalized stylometric features**.
   - **Logistic Regression** is selected as the primary classifier because it yields strong cross-validated F1/ROC-AUC alongside calibrated probabilities and direct per-feature log-odds explanations.

---

## 📁 Project Structure

```text
project 3/
├── app.py                         # Main Streamlit application & multi-page UI
├── requirements.txt               # Python dependencies
├── README.md                      # Setup, VS Code guide, ML documentation
├── .streamlit/
│   └── config.toml                # Streamlit light theme configuration
├── data/
│   ├── sample_reviews.csv         # Documented 200-row synthetic benchmark dataset
│   └── bulk_test_reviews.csv      # 30-row sample CSV for bulk & duplicate testing
├── models/                        # Auto-generated trained pipeline & metrics JSON
├── src/
│   ├── __init__.py
│   ├── config.py                  # Paths, labels, and threshold constants
│   ├── nlp_preprocessing.py       # Cleaning, stopwords, stylometric transformer, signals
│   ├── ml_model.py                # Model training, comparison, persistence, explainability
│   ├── duplicate_detector.py      # Exact/near-duplicate cosine similarity & n-gram detector
│   ├── database.py                # SQLite persistence layer
│   └── ui_components.py           # Custom CSS, KPI cards, Plotly & Matplotlib charts
└── tests/
    └── test_reviewshield.py       # Pytest unit & integration test suite
```

---

## 🚀 Setup & Running in VS Code (Beginner-Friendly Guide)

### 1. Open the Project in VS Code
1. Launch **VS Code**.
2. Click **File → Open Folder...** and select this project folder (`project 3`).
3. Open the integrated terminal in VS Code via **Terminal → New Terminal** (`Ctrl + ~`).

### 2. Create & Activate a Virtual Environment
On **Windows (PowerShell)**:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
*(If PowerShell blocks script execution, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or use `.venv\Scripts\python.exe` directly.)*

On **macOS / Linux**:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Select the Python Interpreter in VS Code (Recommended)
1. Press `Ctrl + Shift + P` (or `Cmd + Shift + P` on macOS).
2. Type **Python: Select Interpreter** and choose the `.venv` interpreter (`./.venv/Scripts/python.exe`).

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Run the Streamlit Web Application
```bash
streamlit run app.py
```
Streamlit will open the app automatically in your browser (typically at `http://localhost:8501`). On first launch, ReviewShield trains and caches the ML pipeline from `data/sample_reviews.csv` into `models/`.

### 6. Run Automated Tests
```bash
pytest -v
```
