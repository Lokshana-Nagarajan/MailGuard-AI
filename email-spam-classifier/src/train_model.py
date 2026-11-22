"""
train_model.py
---------------
End-to-end training script:

    load -> clean -> split -> TF-IDF -> train 3 models -> evaluate ->
    select best -> save model + vectorizer + metadata + metrics + charts

Run with:
    python -m src.train_model
or:
    python src/train_model.py

IMPORTANT (no data leakage): the TfidfVectorizer is fit ONLY on the
training split (`vectorizer.fit_transform(X_train)`), then used to
`.transform()` (never re-fit on) the test split. This mirrors how the
vectorizer will be used at prediction time on genuinely unseen text.
"""

from __future__ import annotations

import json
import os
import sys
import time

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

# Allow running as a plain script (python src/train_model.py) as well as a module.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_and_clean_dataset, DatasetError  # noqa: E402
from src.preprocessing import clean_series  # noqa: E402
from src.visualization import (  # noqa: E402
    plot_class_distribution,
    plot_message_length_distribution,
    plot_confusion_matrix,
    plot_model_comparison,
    plot_wordcloud,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")

RANDOM_STATE = 42
TEST_SIZE = 0.2
APP_VERSION = "1.1.0"


def build_candidate_models():
    """
    Three required models. LinearSVC does not natively expose predict_proba,
    so we wrap it in CalibratedClassifierCV (Platt scaling via sigmoid, fit
    with internal cross-validation on the TRAINING data only) so the app can
    show a genuine, documented probability instead of mislabeling raw decision
    scores as probabilities.
    """
    return {
        "Multinomial Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "Linear SVM": CalibratedClassifierCV(
            LinearSVC(random_state=RANDOM_STATE, max_iter=5000), cv=3
        ),
    }


def evaluate_model(model, X_test_vec, y_test, model_name: str):
    y_pred = model.predict(X_test_vec)
    cm = confusion_matrix(y_test, y_pred)
    # cm layout: [[TN, FP], [FN, TP]] since classes are [0=ham, 1=spam]
    tn, fp, fn, tp = cm.ravel()
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "false_positive_rate": false_positive_rate,
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }
    return metrics, y_pred


def select_best_model(results: list[dict]) -> str:
    """
    Selection criterion (documented, not arbitrary):
    rank by F1 score first; among models within 1 percentage point of the
    top F1 score, prefer the one with higher precision (fewer legitimate
    emails wrongly flagged as spam -- a false positive is more costly than
    a missed spam message in a real inbox).
    """
    best_f1 = max(r["f1_score"] for r in results)
    close_contenders = [r for r in results if (best_f1 - r["f1_score"]) <= 0.01]
    winner = max(close_contenders, key=lambda r: r["precision"])
    return winner["model"]


def main():
    print("=" * 70)
    print("AI EMAIL SPAM CLASSIFIER - TRAINING PIPELINE")
    print("=" * 70)

    t0 = time.time()

    # 1-6: Load + clean dataset ------------------------------------------------
    print("\n[1/8] Loading dataset ...")
    try:
        df, stats = load_and_clean_dataset()
    except DatasetError as exc:
        print(f"\nERROR: {exc}\n")
        sys.exit(1)

    print(f"  Total messages after cleaning : {stats.total_messages}")
    print(f"  Spam messages                 : {stats.spam_count}")
    print(f"  Ham messages                  : {stats.ham_count}")
    print(f"  Spam percentage                : {stats.spam_percentage}%")
    print(f"  Duplicates removed             : {stats.duplicates_removed}")
    print(f"  Rows with missing values removed: {stats.missing_values_removed}")

    # 7: NLP preprocessing ------------------------------------------------------
    print("\n[2/8] Preprocessing text (lowercase, tokenize, stopwords, lemmatize) ...")
    df["cleaned_message"] = clean_series(df["message"])
    df = df[df["cleaned_message"].str.len() > 0].reset_index(drop=True)

    # 8: Stratified train/test split --------------------------------------------
    print("\n[3/8] Splitting into train/test (80/20, stratified) ...")
    X_train, X_test, y_train, y_test = train_test_split(
        df["cleaned_message"],
        df["label"],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["label"],
    )
    print(f"  Train size: {len(X_train)}  |  Test size: {len(X_test)}")

    # 9: TF-IDF -- fit ONLY on training data to avoid data leakage ---------------
    print("\n[4/8] Fitting TF-IDF vectorizer on TRAINING data only ...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),   # unigrams + bigrams capture phrases like "free entry"
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        stop_words=None,       # stopwords already removed in preprocessing
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)  # transform only, never re-fit
    print(f"  Vocabulary size: {len(vectorizer.vocabulary_)}")

    # 10-11: Train and evaluate all models ---------------------------------------
    print("\n[5/8] Training and evaluating candidate models ...")
    candidates = build_candidate_models()
    results = []
    trained_models = {}

    for name, model in candidates.items():
        print(f"  Training: {name} ...")
        model.fit(X_train_vec, y_train)
        metrics, _ = evaluate_model(model, X_test_vec, y_test, name)
        results.append(metrics)
        trained_models[name] = model
        print(
            f"    Accuracy={metrics['accuracy']:.4f}  "
            f"Precision={metrics['precision']:.4f}  "
            f"Recall={metrics['recall']:.4f}  "
            f"F1={metrics['f1_score']:.4f}  "
            f"FPR={metrics['false_positive_rate']:.4f}"
        )

    results_df = pd.DataFrame(results)

    # 12: Select the best model ---------------------------------------------------
    print("\n[6/8] Selecting best model (highest F1; ties broken by precision) ...")
    best_model_name = select_best_model(results)
    best_model = trained_models[best_model_name]
    best_metrics = next(r for r in results if r["model"] == best_model_name)
    print(f"  Selected model: {best_model_name}")

    # 13: Save model, vectorizer, metadata, metrics --------------------------------
    print("\n[7/8] Saving model, vectorizer, metadata, and metrics ...")
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    joblib.dump(best_model, os.path.join(MODELS_DIR, "spam_classifier.pkl"))
    joblib.dump(best_model, os.path.join(MODELS_DIR, "spam_classifier.joblib"))
    joblib.dump(vectorizer, os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"))

    metadata = {
        "app_version": APP_VERSION,
        "best_model": best_model_name,
        "metrics": best_metrics,
        "all_model_results": results,
        "dataset_stats": {
            "total_messages": stats.total_messages,
            "spam_count": stats.spam_count,
            "ham_count": stats.ham_count,
            "spam_percentage": stats.spam_percentage,
            "duplicates_removed": stats.duplicates_removed,
        },
        "train_size": len(X_train),
        "test_size": len(X_test),
        "vocabulary_size": len(vectorizer.vocabulary_),
        "random_state": RANDOM_STATE,
        "trained_at_unix": time.time(),
    }
    with open(os.path.join(MODELS_DIR, "model_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    results_df.to_csv(os.path.join(OUTPUTS_DIR, "model_metrics.csv"), index=False)

    # 14: Generate visualizations ---------------------------------------------------
    print("\n[8/8] Generating visualizations ...")
    plot_class_distribution(df)
    plot_message_length_distribution(df)
    y_pred_best = best_model.predict(X_test_vec)
    plot_confusion_matrix(y_test, y_pred_best, best_model_name)
    plot_model_comparison(results_df)
    try:
        spam_msgs = df[df["label"] == 1]["cleaned_message"].tolist()
        ham_msgs = df[df["label"] == 0]["cleaned_message"].tolist()
        plot_wordcloud(spam_msgs, "Most Common Spam Words", os.path.join(OUTPUTS_DIR, "wordcloud_spam.png"))
        plot_wordcloud(ham_msgs, "Most Common Ham Words", os.path.join(OUTPUTS_DIR, "wordcloud_ham.png"))
    except Exception as exc:  # pragma: no cover - wordcloud is optional/cosmetic
        print(f"  (Word cloud generation skipped: {exc})")

    elapsed = time.time() - t0
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(pd.DataFrame(results).to_string(index=False))
    print(f"\nBest model : {best_model_name}")
    print(f"Saved to   : {MODELS_DIR}")
    print(f"Charts to  : {OUTPUTS_DIR}")
    print(f"Total time : {elapsed:.1f}s")


if __name__ == "__main__":
    main()
