"""
predict.py
----------
Reusable, single-source-of-truth prediction API. The Streamlit app
imports and calls `predict_message()` rather than duplicating any ML
logic -- this guarantees the UI and any other future client (CLI,
batch script, tests) always agree on how a message is classified.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import joblib
import numpy as np

from src.analytics import analyze_message, risk_category, MessageStats
from src.explainability import explain_prediction, WordContribution
from src.logger import get_logger
from src.preprocessing import clean_text

logger = get_logger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

MODEL_PATH = os.path.join(MODELS_DIR, "spam_classifier.pkl")
VECTORIZER_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
METADATA_PATH = os.path.join(MODELS_DIR, "model_metadata.json")


class ModelNotTrainedError(Exception):
    """Raised when model/vectorizer files are missing -- i.e. training hasn't run yet."""


@dataclass
class PredictionResult:
    label: str                      # "SPAM" or "HAM"
    is_spam: bool
    confidence: float                # 0-100, confidence in the predicted label
    spam_probability: float          # 0-100, raw P(spam) from the model
    risk_score: float                # 0-100 (== spam_probability; kept separate for clarity)
    risk_category: str
    stats: MessageStats
    top_spam_words: list = field(default_factory=list)
    top_ham_words: list = field(default_factory=list)
    explanation_supported: bool = True
    explanation_text: str = ""
    model_name: str = ""
    error: str | None = None


class _ModelBundle:
    """Lazily-loaded singleton so Streamlit doesn't reload the model on every rerun."""

    _model = None
    _vectorizer = None
    _metadata = None

    @classmethod
    def load(cls):
        if cls._model is None or cls._vectorizer is None:
            if not (os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH)):
                logger.error("Model/vectorizer files missing -- model not trained yet.")
                raise ModelNotTrainedError(
                    "No trained model found.\n\n"
                    f"Expected files:\n  {MODEL_PATH}\n  {VECTORIZER_PATH}\n\n"
                    "Train the model first by running:\n    python src/train_model.py"
                )
            try:
                cls._model = joblib.load(MODEL_PATH)
                cls._vectorizer = joblib.load(VECTORIZER_PATH)
                logger.info("Model and vectorizer loaded successfully.")
            except Exception as exc:  # pragma: no cover
                logger.error("Failed to load saved model files: %s", exc)
                raise ModelNotTrainedError(f"Failed to load saved model files: {exc}") from exc

            if os.path.exists(METADATA_PATH):
                with open(METADATA_PATH) as f:
                    cls._metadata = json.load(f)
            else:
                cls._metadata = {}
        return cls._model, cls._vectorizer, cls._metadata


def is_model_available() -> bool:
    return os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH)


def get_metadata() -> dict:
    _, _, metadata = _ModelBundle.load()
    return metadata


def predict_message(raw_text: str) -> PredictionResult:
    """
    Classify a single raw (unprocessed) message.

    Returns a PredictionResult. Never raises for ordinary bad input
    (empty string, extremely long text) -- those are handled gracefully
    and reflected in the result. Raises ModelNotTrainedError only if the
    model/vectorizer files genuinely don't exist on disk.
    """
    model, vectorizer, metadata = _ModelBundle.load()
    model_name = metadata.get("best_model", "Unknown") if metadata else "Unknown"

    stats = analyze_message(raw_text or "")

    if not raw_text or not raw_text.strip():
        return PredictionResult(
            label="N/A", is_spam=False, confidence=0.0, spam_probability=0.0,
            risk_score=0.0, risk_category="N/A", stats=stats,
            model_name=model_name, error="Empty input: please enter a message to analyze.",
        )

    cleaned = clean_text(raw_text)
    if not cleaned:
        # Message was non-empty but reduced to nothing after cleaning
        # (e.g. pure punctuation/stopwords) -- still attempt a prediction
        # on the raw text's TF-IDF transform (will likely be all-zero vector).
        cleaned = raw_text.lower()

    X = vectorizer.transform([cleaned])

    pred = int(model.predict(X)[0])
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        spam_probability = float(proba[1]) * 100
    else:
        # Should not happen given our training pipeline (LinearSVC is calibrated),
        # but guard anyway rather than mislabeling a decision score as a probability.
        spam_probability = 100.0 if pred == 1 else 0.0

    confidence = spam_probability if pred == 1 else (100 - spam_probability)
    risk = spam_probability  # risk score IS the model's spam probability, documented as such

    top_spam, top_ham, supported = explain_prediction(vectorizer, model, cleaned)

    # Log metadata only -- never the message body (may contain personal data).
    logger.info(
        "prediction label=%s confidence=%.2f risk=%.2f chars=%d",
        "SPAM" if pred == 1 else "HAM", confidence, risk, len(raw_text),
    )

    if supported and (top_spam or top_ham):
        leading = (top_spam[:3] if pred == 1 else top_ham[:3])
        words_str = ", ".join(w.word for w in leading) if leading else "no single dominant word"
        if pred == 1:
            explanation = (
                f"Classified as SPAM mainly because of terms like: {words_str}. "
                "These words are strongly associated with spam in the training data."
            )
        else:
            explanation = (
                f"Classified as HAM (legitimate) -- it reads like normal conversational "
                f"text and lacks strong spam-indicating terms. "
                f"({words_str + ' had the most ham-leaning weight.' if leading else ''})"
            )
    elif not supported:
        explanation = (
            "Word-level explanation is not available for this model type; "
            "only the prediction and confidence are shown."
        )
    else:
        explanation = "No individual word strongly influenced this prediction."

    return PredictionResult(
        label="SPAM" if pred == 1 else "HAM",
        is_spam=bool(pred == 1),
        confidence=round(confidence, 2),
        spam_probability=round(spam_probability, 2),
        risk_score=round(risk, 2),
        risk_category=risk_category(risk),
        stats=stats,
        top_spam_words=top_spam,
        top_ham_words=top_ham,
        explanation_supported=supported,
        explanation_text=explanation,
        model_name=model_name,
    )


def predict_batch(messages: list[str]) -> list[PredictionResult]:
    """Classify a list of messages (used by the batch-analysis UI page)."""
    return [predict_message(m) for m in messages]


if __name__ == "__main__":
    for msg in [
        "Congratulations! You have won a $1000 prize. Click now to claim your reward!",
        "Hi, are we still meeting tomorrow at 10 AM? Please confirm.",
    ]:
        result = predict_message(msg)
        print(f"\nMessage: {msg}")
        print(f"  -> {result.label} (confidence={result.confidence}%, risk={result.risk_score})")
        print(f"  Explanation: {result.explanation_text}")
