# MailGuard AI -- AI Email Spam Classifier
# Multi-stage build: install dependencies, pre-fetch NLTK data and train the
# model at build time, then ship a slim runtime image.

FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps needed by matplotlib/wordcloud (fonts) -- kept minimal.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download NLTK corpora at build time so the container never needs
# network access at runtime.
RUN python -c "import nltk; [nltk.download(p, quiet=True) for p in ['punkt','punkt_tab','stopwords','wordnet','omw-1.4']]"

COPY . .

# Train the model at image-build time if a dataset is present. If data/spam.csv
# is added later via a volume mount, run `docker exec <container> python src/train_model.py`
# to (re)train after the container starts.
RUN python src/train_model.py || echo "No dataset found at build time -- train after mounting data/spam.csv"

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
