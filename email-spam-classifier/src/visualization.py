"""
visualization.py
-----------------
Chart-generation helpers used both by the offline training script
(saving PNGs into outputs/) and, where noted, by the Streamlit app
(which prefers interactive Plotly for the live dashboard).
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")  # headless-safe backend for script/CLI use
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

sns.set_theme(style="whitegrid")

OUTPUTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs"
)


def _ensure_outputs_dir():
    os.makedirs(OUTPUTS_DIR, exist_ok=True)


def plot_class_distribution(df: pd.DataFrame, save_path: str | None = None):
    _ensure_outputs_dir()
    save_path = save_path or os.path.join(OUTPUTS_DIR, "class_distribution.png")
    counts = df["label"].map({0: "Ham", 1: "Spam"}).value_counts()

    fig, ax = plt.subplots(figsize=(6, 4.5))
    colors = ["#2E8B57" if c == "Ham" else "#C0392B" for c in counts.index]
    ax.bar(counts.index, counts.values, color=colors)
    ax.set_title("Spam vs Ham Distribution")
    ax.set_ylabel("Number of Messages")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 5, str(v), ha="center", fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def plot_message_length_distribution(df: pd.DataFrame, save_path: str | None = None):
    _ensure_outputs_dir()
    save_path = save_path or os.path.join(OUTPUTS_DIR, "message_length_distribution.png")
    lengths = df.assign(length=df["message"].str.len())

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label, name, color in [(0, "Ham", "#2E8B57"), (1, "Spam", "#C0392B")]:
        subset = lengths[lengths["label"] == label]["length"]
        ax.hist(subset, bins=40, alpha=0.6, label=name, color=color, range=(0, 300))
    ax.set_title("Message Length Distribution (Spam vs Ham)")
    ax.set_xlabel("Character Count")
    ax.set_ylabel("Frequency")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def plot_confusion_matrix(y_true, y_pred, model_name: str, save_path: str | None = None):
    _ensure_outputs_dir()
    save_path = save_path or os.path.join(OUTPUTS_DIR, "confusion_matrix.png")
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(5, 4.5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", cbar=False,
        xticklabels=["Ham", "Spam"], yticklabels=["Ham", "Spam"], ax=ax,
    )
    ax.set_title(f"Confusion Matrix - {model_name}")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def plot_model_comparison(results_df: pd.DataFrame, save_path: str | None = None):
    _ensure_outputs_dir()
    save_path = save_path or os.path.join(OUTPUTS_DIR, "model_comparison.png")

    metrics = ["accuracy", "precision", "recall", "f1_score"]
    x = np.arange(len(results_df))
    width = 0.2

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, metric in enumerate(metrics):
        ax.bar(x + i * width, results_df[metric], width, label=metric.replace("_", " ").title())

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(results_df["model"], rotation=15)
    ax.set_ylim(0, 1.05)
    ax.set_title("Model Comparison")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def plot_wordcloud(messages: list[str], title: str, save_path: str):
    """Generate a word cloud image from a list of already-cleaned messages."""
    from wordcloud import WordCloud

    _ensure_outputs_dir()
    text = " ".join(messages) if messages else "no_data"
    wc = WordCloud(width=800, height=400, background_color="white", colormap="viridis").generate(text)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path
