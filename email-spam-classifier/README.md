<p align="center">
  <img src="assets/logo_128.png" alt="MailGuard AI logo" width="110"/>
</p>

# MailGuard AI -- AI-Based Email Spam Classifier Using NLP and Machine Learning

A complete, working, presentation-ready machine learning application that
classifies an email/SMS message as **SPAM** or **HAM (legitimate)** using
classical NLP preprocessing, TF-IDF feature extraction, and supervised
machine learning -- wrapped in an interactive Streamlit dashboard.

> A complete, containerized, CI-tested application. Like every spam filter, it
> analyzes message text only and cannot guarantee detection of all spam or
> phishing -- see [Security notice & limitations](#18-limitations).

---

## 1. Overview

The app analyzes the **text** of a message and predicts whether it is spam,
along with a confidence score, a 0-100 spam-risk gauge, message-content
statistics, and a plain-English explanation of *which words* drove the
decision -- grounded in the actual trained model's weights, not a
hand-written keyword list.

## 2. Problem statement

Spam/phishing messages are a constant nuisance and security risk; manual
filtering doesn't scale. This project evaluates whether a supervised
ML model can learn to separate spam from legitimate messages using only
message text.

## 3. Objectives

- Build a reproducible NLP + ML pipeline with **no data leakage**.
- Compare multiple classical ML models on standard metrics and pick the
  best one using a documented, defensible rule.
- Make predictions **explainable** (which words mattered, and why).
- Ship a polished, usable Streamlit interface, including single-message
  and batch (CSV) analysis.

## 4. Features

| Area | What it does |
|---|---|
| Spam Detection | Paste a message, get SPAM/HAM + confidence + risk gauge |
| Explainability | Shows the actual words that pushed the score toward spam/ham |
| Content analysis | URL count, uppercase words, digits, repeated punctuation, urgency words |
| Batch analysis | Upload a CSV, classify every row, download results |
| Model performance | Accuracy/Precision/Recall/F1/FPR for all 3 trained models |
| Dataset insights | Class balance, message-length distributions, word clouds |
| About | Full write-up of methodology, architecture, and limitations |

## 5. Technologies

Python 3 · scikit-learn · pandas / numpy · NLTK · Streamlit · Plotly ·
Matplotlib / Seaborn · joblib · WordCloud

## 6. System architecture / ML workflow

```
Email/Message Input
        |
Text Cleaning & NLP Preprocessing (NLTK: lowercase, tokenize,
        |                          stopword removal, lemmatization)
Feature Extraction (TF-IDF, unigrams+bigrams, fit on TRAIN split only)
        |
Train & Compare Models (Multinomial NB / Logistic Regression / Linear SVM)
        |
Evaluate (Accuracy, Precision, Recall, F1, False-Positive Rate)
        |
Select Best Model (highest F1; ties within 1pt broken by precision)
        |
Save Model + Vectorizer (joblib) + Metadata (json)
        |
Real-time Prediction (src/predict.py) -> Confidence + Risk Score
        |
Explainability (model coefficients / NB log-odds -> influential words)
        |
Streamlit Dashboard
```

## 7. Dataset

**UCI SMS Spam Collection** (Kaggle mirror:
`uciml/sms-spam-collection-dataset`) -- 5,574 raw rows of English SMS
messages labeled `ham` or `spam`, columns `v1` (label) and `v2` (text).

Expected location:
```
data/spam.csv
```
If the file is missing, `src/data_loader.py` raises a clear error
explaining exactly where to put it and where to download it from --
it never fabricates training data.

**Note:** this CSV ships with 3 extra unnamed "ghost" columns and a
Latin-1 encoding, both handled automatically by the loader.

## 8. Project structure

```
email-spam-classifier/
├── app.py                      # Streamlit UI (imports src/, no duplicated ML logic)
├── requirements.txt
├── README.md
├── .gitignore
├── Dockerfile / docker-compose.yml / .dockerignore
├── Procfile / runtime.txt
├── CHANGELOG.md / VERSION
├── .streamlit/config.toml      # theme + server config
├── .github/workflows/ci.yml    # CI pipeline
├── assets/                     # logo + favicon
├── data/
│   └── spam.csv                # UCI SMS Spam Collection (place here)
├── models/
│   ├── spam_classifier.pkl      # best trained model (joblib)
│   ├── spam_classifier.joblib   # same model, alternate extension
│   ├── tfidf_vectorizer.pkl     # fitted TF-IDF vectorizer
│   └── model_metadata.json      # metrics, dataset stats, chosen model
├── src/
│   ├── data_loader.py           # load/validate/clean the dataset
│   ├── preprocessing.py         # NLP cleaning (shared train/predict)
│   ├── train_model.py           # full training pipeline (entry point)
│   ├── predict.py                # single source of truth for predictions
│   ├── explainability.py         # model-grounded word-level explanations
│   ├── analytics.py              # text statistics + risk categories
│   ├── logger.py                 # rotating-file + stdout logging
│   └── visualization.py          # matplotlib/seaborn chart generation
├── tests/
│   ├── test_preprocessing.py
│   ├── test_data_loader.py
│   └── test_prediction.py
├── notebooks/
│   └── exploratory_analysis.ipynb
└── outputs/                     # generated charts + metrics.csv (training output)
```

## 9. Installation

Run these one at a time.

**Windows (PowerShell):**

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If PowerShell blocks activation ("running scripts is disabled"), run once per window:
`Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned`

**macOS / Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

NLTK resources (punkt, stopwords, wordnet) are downloaded automatically
on first run if missing -- no manual step needed.

## 10. How to run

### Step 1 -- Train the model (required once, or after changing the dataset)

```bash
python src/train_model.py
```

This loads `data/spam.csv`, cleans it, trains and compares 3 models,
selects the best one, and writes `models/*.pkl`, `models/model_metadata.json`,
`outputs/model_metrics.csv`, and all chart PNGs in `outputs/`.

### Step 2 -- Launch the dashboard

```bash
streamlit run app.py
```

Open the URL it prints (typically `http://localhost:8501`).

### Step 3 (optional) -- Run the automated tests

```bash
pytest tests/ -v
```

## 11. Actual evaluation metrics (this run)

Trained on 5,158 cleaned/deduplicated messages (642 spam / 4,516 ham,
80/20 stratified split -> 4,123 train / 1,031 test). TF-IDF: 5,000
features, unigrams + bigrams, fit on the training split only.

| Model | Accuracy | Precision | Recall | F1 Score | False Positive Rate |
|---|---|---|---|---|---|
| Multinomial Naive Bayes | 97.09% | 100.00% | 76.56% | 86.73% | 0.00% |
| Logistic Regression | 96.22% | 95.88% | 72.66% | 82.67% | 0.44% |
| **Linear SVM (selected)** | **98.55%** | **96.69%** | **91.41%** | **93.98%** | **0.44%** |

**Selected model: Linear SVM** (wrapped in `CalibratedClassifierCV` for
genuine probability estimates). Selection rule: highest F1 score; ties
within 1 percentage point broken in favor of higher precision, since a
false positive (legitimate email marked spam) is more costly in practice
than a missed spam message.

These numbers come directly from this project's own `outputs/model_metrics.csv`
after an actual training run -- they are not estimated or assumed.

## 12. Example predictions (verified against the trained model)

| Message | Prediction | Confidence |
|---|---|---|
| "Congratulations! You have won a $1000 prize. Click now to claim your reward!" | SPAM | 98.4% |
| "Hi, are we still meeting tomorrow at 10 AM? Please confirm." | HAM | 99.9% |
| "FREE FREE FREE!!! WIN CASH NOW CLICK http://scam.xyz !!!" | SPAM | ~100% |
| "See you at the office at 9am." | HAM | 99.7% |
| "" (empty input) | N/A | handled gracefully with a message, no crash |

## 13. Explainability

For Logistic Regression / Linear SVM, a word's contribution to a specific
prediction is `TF-IDF value of that word in the message x model coefficient
for that word`. For Multinomial Naive Bayes, it's the log-odds
`log P(word | spam) - log P(word | ham)`. Both are read directly from the
fitted model -- there is no hardcoded spam-keyword list anywhere in the
classification path.

## 14. Batch analysis

Upload any CSV with a text column (auto-detected from common names like
`v2`, `text`, `message`; otherwise pick manually). Every row gets a
prediction, confidence, risk score, and risk category, and the full
result table can be downloaded as a new CSV.

## 15. Testing

28 automated pytest tests cover: text preprocessing edge cases, dataset
loading/validation (including a deliberately malformed CSV), label
mapping, obvious-spam / obvious-ham classification, empty / whitespace /
very short / very long input handling, batch prediction, and confidence
and risk-score bounds. All 28 passed in this project's own test run.

The Streamlit app itself was additionally smoke-tested with Streamlit's
`AppTest` framework across all 7 pages (Home, Spam Detection, Batch
Analysis, Model Performance, Dataset Insights, Explainability, About) and
the Analyze-Email button flow, with zero runtime exceptions.

## 16. Deployment

### Option A -- Docker (recommended)

```bash
docker compose up --build
```

Open `http://localhost:8501`. The image pre-downloads NLTK data and trains
the model at build time; `data/`, `models/`, and `outputs/` are mounted as
volumes so you can swap the dataset and retrain without rebuilding:

```bash
docker exec mailguard-ai python src/train_model.py
```

A container healthcheck polls Streamlit's `/_stcore/health` endpoint.

### Option B -- Streamlit Community Cloud (free)

1. Push this repo to GitHub (include `data/spam.csv` and the `models/` folder, or
   let the first run train: run `python src/train_model.py` locally and commit `models/`).
2. At share.streamlit.io choose the repo, branch `main`, main file `app.py`.
3. Deploy. `requirements.txt` is picked up automatically.

### Option C -- Render / Heroku-style PaaS

`Procfile` and `runtime.txt` are included. Create a new Web Service from the
repo; the `release` step trains the model and the `web` step starts Streamlit
bound to `$PORT`.

### Continuous integration

`.github/workflows/ci.yml` runs on every push/PR: installs dependencies,
trains the model, runs all pytest tests, smoke-tests the Streamlit app, and
builds the Docker image.

### Operations

- Logs go to stdout (captured by `docker logs` / PaaS log streams) and to a
  rotating `logs/app.log`. Only prediction metadata (label, confidence, length)
  is logged -- never message text.
- Configuration (theme, upload limit, CORS/XSRF) lives in `.streamlit/config.toml`.
- Retraining: replace `data/spam.csv`, run `python src/train_model.py`, restart.

### Publishing to GitHub

```powershell
git init
git add .
git commit -m "Initial commit: MailGuard AI"
git branch -M main
git remote add origin https://github.com/<your-username>/mailguard-ai.git
git push -u origin main
```

Create the empty repository on github.com first (do not add a README, .gitignore or licence there).
Later updates: `git add .`, `git commit -m "message"`, `git push`.

## 17. Screenshots

*(Add screenshots of the running app here for your presentation --
Home, Spam Detection result, Model Performance, and Dataset Insights
pages are good choices.)*

```
outputs/screenshot_home.png
outputs/screenshot_spam_detection.png
outputs/screenshot_model_performance.png
```

## 18. Limitations

- Trained on **SMS** messages, not full emails -- email-specific phishing
  patterns (spoofed display names, malicious attachments, lookalike
  domains) are not directly represented in this training data.
- Analyzes **text content only**. It does **not** evaluate sender
  reputation, email headers, attachments, embedded tracking pixels, or
  authentication mechanisms (SPF/DKIM/DMARC) -- all of which a real-world
  email security system also uses.
- Like any statistical model, it can misclassify unusual, adversarial, or
  out-of-distribution messages. It is **not** a guarantee of detecting
  every spam or phishing attempt.
- Calibrated probabilities for the Linear SVM are produced via
  `CalibratedClassifierCV` (Platt scaling) -- a well-documented
  approximation, not an exact posterior probability.

## 19. Future enhancements

- Fine-tune on a labeled **email** (not just SMS) corpus for better
  real-world transfer.
- Incorporate header/metadata features (sender domain age, SPF/DKIM
  results, reply-to mismatches).
- Add transformer-based embeddings (e.g. a distilled BERT) as a 4th model
  for comparison.
- Add a feedback loop so users can correct mispredictions and the model
  can be periodically retrained.

## 20. Conclusion

MailGuard AI delivers a complete, leakage-free, explainable NLP/ML pipeline
for spam detection, end to end: raw CSV -> trained, evaluated, saved model ->
branded dashboard with single-message and batch prediction -> containerized,
CI-tested deployment. It is one layer of defense and should be combined with
sender-reputation, header, URL, and attachment checks for full email security.
