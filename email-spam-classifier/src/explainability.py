"""
explainability.py
------------------
Turns a trained (linear-ish) model + TF-IDF vectorizer into a list of
"this word pushed the prediction toward SPAM / HAM" contributions for
a single message.

This is grounded in real model internals, not a hardcoded keyword list:

- LogisticRegression / LinearSVC: both expose `coef_`, a weight per
  TF-IDF feature. A word's contribution to THIS message is
  (tfidf_value_of_word_in_message * coef_of_word). Positive values
  push toward spam (class 1), negative values push toward ham.
- MultinomialNB: has no linear `coef_`, but `feature_log_prob_` gives
  log P(word | class). We use the log-odds
  log P(word|spam) - log P(word|ham) as the per-word contribution,
  which is the natural analogue for a Naive Bayes model.

If the active model type supports neither mechanism, we say so plainly
rather than inventing an explanation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB


@dataclass
class WordContribution:
    word: str
    contribution: float  # positive = pushes toward SPAM, negative = pushes toward HAM


def explain_prediction(vectorizer, model, cleaned_text: str, top_n: int = 10):
    """
    Returns (top_spam_words, top_ham_words, supported: bool).

    `model` may be the raw estimator, or a CalibratedClassifierCV wrapping
    a LinearSVC (used so LinearSVC can expose predict_proba) -- in that
    case we reach into `.calibrated_classifiers_[0].estimator` to recover
    the underlying linear model's coefficients.
    """
    vec = vectorizer.transform([cleaned_text])
    feature_names = np.array(vectorizer.get_feature_names_out())
    active_idx = vec.nonzero()[1]

    if len(active_idx) == 0:
        return [], [], True  # nothing to explain, but mechanism is supported

    underlying = _unwrap_calibrated(model)

    if isinstance(underlying, (LogisticRegression, LinearSVC)):
        coefs = underlying.coef_[0]
        tfidf_values = np.asarray(vec[:, active_idx].todense()).flatten()
        contributions = coefs[active_idx] * tfidf_values
    elif isinstance(underlying, MultinomialNB):
        # log P(word|spam) - log P(word|ham); class order is [0=ham, 1=spam]
        log_prob_ham = underlying.feature_log_prob_[0]
        log_prob_spam = underlying.feature_log_prob_[1]
        contributions = (log_prob_spam - log_prob_ham)[active_idx]
    else:
        return [], [], False

    words = feature_names[active_idx]
    pairs = sorted(zip(words, contributions), key=lambda p: p[1], reverse=True)

    top_spam = [WordContribution(w, float(c)) for w, c in pairs if c > 0][:top_n]
    top_ham = [WordContribution(w, float(c)) for w, c in pairs if c < 0][:top_n]
    top_ham.sort(key=lambda wc: wc.contribution)  # most negative (most ham-like) first

    return top_spam, top_ham, True


def _unwrap_calibrated(model):
    """Best-effort unwrap of a CalibratedClassifierCV to its base estimator."""
    if hasattr(model, "calibrated_classifiers_"):
        try:
            return model.calibrated_classifiers_[0].estimator
        except (AttributeError, IndexError):
            return model
    return model


def global_top_features(vectorizer, model, top_n: int = 15):
    """
    Dataset-wide (not message-specific) most spam-indicative and most
    ham-indicative features, derived from the trained model's own
    learned weights -- used on the 'Important Keywords' / 'Explainability'
    pages. This is NOT a hardcoded list; it is read directly off the
    fitted model.
    """
    feature_names = np.array(vectorizer.get_feature_names_out())
    underlying = _unwrap_calibrated(model)

    if isinstance(underlying, (LogisticRegression, LinearSVC)):
        coefs = underlying.coef_[0]
    elif isinstance(underlying, MultinomialNB):
        coefs = underlying.feature_log_prob_[1] - underlying.feature_log_prob_[0]
    else:
        return [], []

    order = np.argsort(coefs)
    top_ham_idx = order[:top_n]
    top_spam_idx = order[::-1][:top_n]

    top_spam = [(feature_names[i], float(coefs[i])) for i in top_spam_idx]
    top_ham = [(feature_names[i], float(coefs[i])) for i in top_ham_idx]
    return top_spam, top_ham


# ---------------------------------------------------------------------------
# Message highlighting (used by the UI to colour influential words)
# ---------------------------------------------------------------------------
import html as _html
import re as _re
from functools import lru_cache as _lru_cache

from src.preprocessing import clean_text as _clean_text


@_lru_cache(maxsize=4096)
def _normalize_token(token: str) -> str:
    return _clean_text(token)


def highlight_html(text: str, top_spam, top_ham) -> str:
    """
    Return HTML of `text` where words that the model found influential are
    wrapped in <mark> tags: red = pushed toward SPAM, green = toward HAM.
    Intensity scales with the size of the contribution. All user text is
    HTML-escaped, so this is safe to render with unsafe_allow_html.
    """
    weights: dict[str, float] = {}
    for wc in list(top_spam) + list(top_ham):
        for part in wc.word.split():  # bigrams contribute to both words
            if abs(wc.contribution) > abs(weights.get(part, 0.0)):
                weights[part] = wc.contribution

    if not weights:
        return _html.escape(text).replace("\n", "<br>")

    max_abs = max(abs(v) for v in weights.values()) or 1.0
    out = []
    for piece in _re.findall(r"\w+|\W+", text):
        if _re.match(r"\w", piece):
            weight = weights.get(_normalize_token(piece))
            if weight:
                alpha = 0.25 + 0.55 * min(abs(weight) / max_abs, 1.0)
                rgb = "231,76,60" if weight > 0 else "46,204,113"
                out.append(
                    f'<mark style="background:rgba({rgb},{alpha:.2f});padding:1px 5px;'
                    f'border-radius:5px;color:inherit;" title="{weight:+.3f}">'
                    f"{_html.escape(piece)}</mark>"
                )
                continue
        out.append(_html.escape(piece))
    return "".join(out).replace("\n", "<br>")
