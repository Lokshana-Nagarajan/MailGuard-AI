"""
preprocessing.py
-----------------
Text-cleaning utilities used identically at training time and at
prediction time (this consistency is what prevents "train/serve skew",
a very common bug in real NLP systems).

Design choice: we do NOT strip digits, currency symbols, or punctuation
entirely, because spam messages lean heavily on patterns like
"WIN $$$ 1000!!!" or "call 09061701461" -- those characters are useful
signals, not noise. We only remove things that add no information
(HTML tags, extra whitespace, stopwords) and normalize case.
"""

from __future__ import annotations

import re
import string

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


def ensure_nltk_resources() -> None:
    """Download the NLTK resources we need if they are not already present.

    Safe to call repeatedly; nltk.download() is a no-op if the resource
    already exists locally (it just checks the cache first).
    """
    resources = {
        "tokenizers/punkt": "punkt",
        "tokenizers/punkt_tab": "punkt_tab",
        "corpora/stopwords": "stopwords",
        "corpora/wordnet": "wordnet",
        "corpora/omw-1.4": "omw-1.4",
    }
    for path, package in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(package, quiet=True)


ensure_nltk_resources()

_STOPWORDS = set(stopwords.words("english"))
# Keep a few words that carry spam/ham signal even though NLTK treats
# them as stopwords (e.g. negations can flip meaning).
_KEEP_WORDS = {"not", "no", "free", "win", "now"}
_STOPWORDS -= _KEEP_WORDS

_LEMMATIZER = WordNetLemmatizer()

_URL_RE = re.compile(r"(https?://\S+|www\.\S+)")
_HTML_RE = re.compile(r"<[^>]+>")
_EXTRA_WHITESPACE_RE = re.compile(r"\s+")
# Punctuation to strip, but keep $, !, and digits -- they're useful spam signals.
_PUNCT_TO_STRIP = "".join(c for c in string.punctuation if c not in "$!")


def clean_text(text: str, use_lemmatization: bool = True) -> str:
    """
    Clean and normalize a single message for TF-IDF feature extraction.

    Steps: lowercase -> strip HTML -> mark URLs -> normalize whitespace
    -> tokenize -> drop stopwords/pure punctuation -> (optional) lemmatize.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    text = text.lower()
    text = _HTML_RE.sub(" ", text)
    # Replace URLs with a single token rather than deleting them outright --
    # "message contains a link" is itself a useful spam signal.
    text = _URL_RE.sub(" urltoken ", text)
    text = _EXTRA_WHITESPACE_RE.sub(" ", text).strip()

    tokens = word_tokenize(text)

    cleaned_tokens = []
    for tok in tokens:
        stripped = tok.strip(_PUNCT_TO_STRIP)
        if not stripped:
            continue
        if stripped in _STOPWORDS:
            continue
        if use_lemmatization:
            stripped = _LEMMATIZER.lemmatize(stripped)
        cleaned_tokens.append(stripped)

    return " ".join(cleaned_tokens)


def clean_series(messages) -> list:
    """Vectorized helper: clean an iterable (e.g. a pandas Series) of messages."""
    return [clean_text(m) for m in messages]


if __name__ == "__main__":
    samples = [
        "Congratulations! You WON a $1000 prize!! Click http://bit.ly/x now!!!",
        "Hi, are we still meeting tomorrow at 10 AM? Please confirm.",
    ]
    for s in samples:
        print(repr(s), "->", repr(clean_text(s)))
