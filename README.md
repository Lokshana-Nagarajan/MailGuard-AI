# MailGuard-AI
<p align="center">
  <img src="assets/logo_128.png" alt="MailGuard AI logo" width="110"/>
</p>

<h1 align="center">MailGuard AI</h1>

<p align="center">
  Intelligent email &amp; SMS spam detection, powered by NLP and machine learning.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11-blue" alt="Python 3.11"/>
  <img src="https://img.shields.io/badge/scikit--learn-ML-orange" alt="scikit-learn"/>
  <img src="https://img.shields.io/badge/Streamlit-dashboard-red" alt="Streamlit"/>
  <img src="https://img.shields.io/badge/tests-28%20passing-brightgreen" alt="28 tests"/>
</p>

---

## Overview

MailGuard AI classifies a message as **SPAM** or **HAM (legitimate)** using a supervised
machine-learning model trained on the UCI SMS Spam Collection. For every message it returns
a verdict, a confidence score, a 0-100 spam-risk gauge, and an explanation of **which words
drove the decision**, taken from the trained model's own weights rather than a hand-written
keyword list.

## Features

- **Spam detection:** paste a message and get an instant verdict with confidence and risk level.
- **Explainable results:** influential words are highlighted inside the message (red = spam, green = legitimate).
- **Batch analysis:** upload a CSV, classify up to 5,000 messages, and download the results.
- **Model comparison:** Multinomial Naive Bayes, Logistic Regression and a calibrated Linear SVM are compared automatically.
- **Dashboards:** model performance, confusion matrix, dataset insights, and word frequencies.
- **Leakage-free pipeline:** stratified split, TF-IDF fitted on training data only.
- **Deployment ready:** Dockerfile, docker-compose, Procfile, and a GitHub Actions CI workflow.
- **Light and dark mode** support.

## How it works

```
Message -> Clean & tokenize -> TF-IDF features -> ML model -> Verdict + risk score -> Word-level explanation
```

1. **Preprocessing (NLTK):** lowercasing, HTML/URL normalization, tokenization, stopword removal, lemmatization. Signal words such as "free", "win" and "not" are kept.
2. **Features:** TF-IDF over unigrams and bigrams (5,000 features).
3. **Models:** three classifiers are trained and evaluated on held-out data.
4. **Selection:** highest F1 score; ties within one point are broken by precision, because wrongly flagging a real email costs more than missing one spam message.
5. **Prediction:** the saved model returns a calibrated probability, which becomes the risk score.

## Results

Trained on 5,158 cleaned, de-duplicated messages (642 spam, 4,516 ham), with an 80/20 stratified split.

| Model | Accuracy | Precision | Recall | F1 | False-positive rate |
|---|---|---|---|---|---|
| Multinomial Naive Bayes | 97.09% | 100.00% | 76.56% | 86.73% | 0.00% |
| Logistic Regression | 96.22% | 95.88% | 72.66% | 82.67% | 0.44% |
| **Linear SVM (selected)** | **98.55%** | **96.69%** | **91.41%** | **93.98%** | **0.44%** |

These numbers come from `outputs/model_metrics.csv` after an actual training run.

## Screenshots

<!-- Add your screenshots here, for example:
![Spam detection](outputs/screenshot_spam_detection.png)
![Model performance](outputs/screenshot_model_performance.png)
-->

## Quick start

Run these from the project folder, one at a time.

**Windows (PowerShell)**

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
python src/train_model.py
streamlit run app.py
```

**macOS / Linux**

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python src/train_model.py
streamlit run app.py
```

Open `http://localhost:8501`. NLTK data downloads automatically on first run.

**Run the tests**

```bash
pytest tests/ -v
```

## Docker

```bash
docker compose up --build
```

## Project structure

```
mailguard-ai/
├── app.py                  # Streamlit UI
├── requirements.txt
├── Dockerfile / docker-compose.yml
├── .github/workflows/ci.yml
├── .streamlit/config.toml
├── assets/                 # logo and favicon
├── data/spam.csv           # UCI SMS Spam Collection
├── models/                 # trained model, vectorizer, metadata
├── outputs/                # charts and metrics
├── src/
│   ├── data_loader.py      # load, validate and clean the dataset
│   ├── preprocessing.py    # NLP cleaning (shared by training and prediction)
│   ├── train_model.py      # full training pipeline
│   ├── predict.py          # prediction API
│   ├── explainability.py   # word contributions and highlighting
│   ├── analytics.py        # message statistics and risk categories
│   ├── batch.py            # batch CSV analysis
│   ├── visualization.py    # chart generation
│   └── logger.py           # logging
└── tests/                  # 28 pytest tests
```

## Dataset

[UCI SMS Spam Collection](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
(5,572 labelled English SMS messages). Place it at `data/spam.csv`. If the file is missing,
the app shows where to get it and never generates fake training data.

## Limitations and security notice

MailGuard AI analyzes **message text only**. It is one layer of defense, not a complete
email-security system, and it cannot guarantee detection of every spam or phishing message.

- It is trained on SMS data, so email-specific phishing (spoofed senders, malicious attachments, lookalike domains) is not directly represented.
- It does not inspect sender reputation, headers, URLs, attachments, or SPF/DKIM/DMARC results.
- Messages that read like normal conversation but are really promotions can be missed.
- Linear SVM probabilities come from Platt-scaling calibration, which is an approximation.

## Roadmap

- Train on a labelled email corpus
- Add sender and header signals
- Add a transformer-based model for comparison
- Add a user-feedback loop for retraining

## Tech stack

Python, scikit-learn, NLTK, pandas, NumPy, Streamlit, Plotly, Matplotlib, Seaborn, joblib, Docker, GitHub Actions.

## License

Add a license of your choice (MIT is a common default) before sharing publicly.
