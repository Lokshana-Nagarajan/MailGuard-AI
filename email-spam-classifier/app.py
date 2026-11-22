"""
app.py -- MailGuard AI (Streamlit UI)

UI only: all ML logic lives in src/ and is imported, never duplicated.
Run with:  streamlit run app.py
"""

from __future__ import annotations

import base64
import html
import os
import time
from collections import Counter

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from src.batch import MAX_BATCH_ROWS, analyze_dataframe, detect_text_column
from src.data_loader import DatasetError, load_and_clean_dataset
from src.explainability import global_top_features, highlight_html
from src.predict import ModelNotTrainedError, get_metadata, is_model_available, predict_message
from src.preprocessing import clean_series, clean_text

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
ASSETS_DIR = os.path.join(PROJECT_ROOT, "assets")

APP_NAME = "MailGuard AI"
APP_TAGLINE = "Intelligent email & SMS spam detection, powered by NLP and machine learning"

PAGES = [
    "🏠 Home",
    "🔍 Spam Detection",
    "📂 Batch Analysis",
    "📊 Model Performance",
    "📈 Dataset Insights",
    "🧠 Explainability",
    "🛡️ Security & About",
]

EXAMPLES = {
    "Prize notification": "Congratulations! You have won a $1000 prize. Click now to claim your reward!",
    "Account alert": "URGENT! Your account has been suspended. Verify now at http://secure-login.example to avoid closure!",
    "Competition entry": "FREE entry in our weekly competition to win tickets! Text WIN to 80086 now. T&Cs apply.",
    "Meeting confirmation": "Hi, are we still meeting tomorrow at 10 AM? Please confirm.",
    "Casual chat": "Running late, will be there in 15 mins. Save me a seat!",
}
EXAMPLE_PLACEHOLDER = "Choose an example message..."

_favicon = os.path.join(ASSETS_DIR, "favicon.png")
st.set_page_config(
    page_title=APP_NAME,
    page_icon=Image.open(_favicon) if os.path.exists(_favicon) else "📧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# Styling. Colours use translucent greys / inherit so it works in light AND dark mode.
# ----------------------------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
.stApp { font-family: 'Inter', 'Segoe UI', system-ui, sans-serif; }
.block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1180px; }
footer { visibility: hidden; }

.hero { background: linear-gradient(135deg,#1a1a2e 0%,#16213e 45%,#11998e 100%); border-radius: 20px;
        padding: 2rem 2.2rem; color: #fff; display: flex; gap: 1.6rem; align-items: center;
        box-shadow: 0 10px 30px rgba(17,153,142,.18); margin-bottom: 1.4rem; }
.hero img { width: 92px; height: 92px; flex: none; }
.hero-title { font-size: 2.3rem; font-weight: 800; letter-spacing: -.02em; line-height: 1.1; }
.hero-sub { font-size: 1.05rem; opacity: .92; margin-top: .45rem; }
.hero-chips { margin-top: .9rem; display: flex; flex-wrap: wrap; gap: .45rem; }
.hero-chips span { background: rgba(255,255,255,.16); border: 1px solid rgba(255,255,255,.28);
                   padding: .2rem .7rem; border-radius: 999px; font-size: .78rem; font-weight: 600; }

.page-head { margin-bottom: 1.2rem; padding-bottom: .7rem; border-bottom: 3px solid #11998e; }
.page-title { font-size: 1.9rem; font-weight: 800; letter-spacing: -.01em; }
.page-sub { opacity: .7; margin-top: .15rem; }

.mg-card { background: rgba(128,128,128,.07); border: 1px solid rgba(128,128,128,.22); border-radius: 14px;
           padding: 1rem 1.1rem; height: 100%; transition: transform .15s, box-shadow .15s; }
.mg-card:hover { transform: translateY(-2px); box-shadow: 0 6px 18px rgba(0,0,0,.10); }
.mg-card-label { font-size: .72rem; text-transform: uppercase; letter-spacing: .07em; opacity: .65; font-weight: 600; }
.mg-card-value { font-size: 1.5rem; font-weight: 700; margin-top: .15rem; line-height: 1.2; }
.mg-card-sub { font-size: .8rem; opacity: .65; margin-top: .2rem; }
.mg-feature { font-size: .92rem; opacity: .85; margin-top: .35rem; line-height: 1.5; }
.mg-feature-title { font-weight: 700; font-size: 1.05rem; }

.result-banner { border-radius: 16px; padding: 1.25rem 1.6rem; color: #fff; display: flex; align-items: center;
                 gap: 1rem; margin: .4rem 0 1rem 0; box-shadow: 0 8px 22px rgba(0,0,0,.14); }
.result-spam { background: linear-gradient(90deg,#ff512f,#dd2476); }
.result-ham { background: linear-gradient(90deg,#11998e,#2fbf71); }
.result-icon { font-size: 2.4rem; }
.result-title { font-size: 1.5rem; font-weight: 800; line-height: 1.15; }
.result-sub { opacity: .93; font-size: .95rem; margin-top: .15rem; }

.hl-box { background: rgba(128,128,128,.07); border: 1px solid rgba(128,128,128,.22); border-radius: 12px;
          padding: 1rem 1.2rem; line-height: 1.95; font-size: 1.02rem; }
.legend { font-size: .82rem; opacity: .8; margin-top: .6rem; }
.legend mark { padding: 1px 8px; border-radius: 5px; color: inherit; }

.flow { display: flex; flex-wrap: wrap; align-items: center; gap: .5rem; }
.flow .step { padding: .5rem .95rem; border-radius: 999px; background: rgba(17,153,142,.14);
              border: 1px solid rgba(17,153,142,.45); font-weight: 600; font-size: .9rem; }
.flow .arrow { opacity: .5; }
.chip { display: inline-block; padding: .2rem .7rem; margin: .15rem .2rem .15rem 0; border-radius: 999px;
        background: rgba(128,128,128,.14); border: 1px solid rgba(128,128,128,.25); font-size: .85rem; }

.stButton > button, .stDownloadButton > button { border-radius: 10px; font-weight: 600; }
.stButton > button[kind="primary"], button[data-testid="stBaseButton-primary"] {
    background: linear-gradient(90deg,#11998e,#0f8579); border: none; color: #fff; }
.stTextArea textarea { border-radius: 12px; font-size: 1rem; }
[data-testid="stSidebar"] { border-right: 1px solid rgba(128,128,128,.2); }
.mg-footer { text-align: center; opacity: .55; font-size: .8rem; margin-top: 3rem; }
</style>
""",
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def _logo_b64() -> str:
    path = os.path.join(ASSETS_DIR, "logo_128.png")
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


@st.cache_data(show_spinner=False)
def _cached_dataset():
    return load_and_clean_dataset()


@st.cache_resource(show_spinner=False)
def _cached_vectorizer_and_model():
    return (
        joblib.load(os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")),
        joblib.load(os.path.join(MODELS_DIR, "spam_classifier.pkl")),
    )


@st.cache_data(show_spinner="Analyzing vocabulary...")
def _top_words(n: int = 20) -> dict:
    df, _ = load_and_clean_dataset()
    out = {}
    for label, name in ((1, "Spam"), (0, "Ham")):
        msgs = clean_series(df[df["label"] == label]["message"])
        out[name] = Counter(w for m in msgs for w in m.split()).most_common(n)
    return out


def card(label: str, value, sub: str = "", accent: str | None = None) -> str:
    style = f' style="border-top:4px solid {accent};"' if accent else ""
    return (
        f'<div class="mg-card"{style}><div class="mg-card-label">{html.escape(str(label))}</div>'
        f'<div class="mg-card-value">{html.escape(str(value))}</div>'
        f'<div class="mg-card-sub">{html.escape(str(sub))}</div></div>'
    )


def page_header(title: str, subtitle: str):
    st.markdown(
        f'<div class="page-head"><div class="page-title">{html.escape(title)}</div>'
        f'<div class="page-sub">{html.escape(subtitle)}</div></div>',
        unsafe_allow_html=True,
    )


def require_model():
    if not is_model_available():
        st.error("No trained model found. Run `python src/train_model.py` first, then refresh this page.")
        st.stop()


def risk_color(score: float) -> str:
    return "#2ecc71" if score <= 30 else "#f1c40f" if score <= 60 else "#e67e22" if score <= 80 else "#e74c3c"


def render_gauge(score: float):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score, number={"suffix": "%"},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": risk_color(score)},
               "steps": [{"range": [0, 30], "color": "rgba(46,204,113,.18)"},
                         {"range": [30, 60], "color": "rgba(241,196,15,.18)"},
                         {"range": [60, 80], "color": "rgba(230,126,34,.18)"},
                         {"range": [80, 100], "color": "rgba(231,76,60,.18)"}]},
        title={"text": "Spam risk score"}))
    fig.update_layout(height=250, margin=dict(t=50, b=5, l=25, r=25), paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, width="stretch")


def go_to(page_name: str):
    st.session_state["nav_page"] = page_name


def load_example():
    choice = st.session_state.get("example_choice")
    if choice in EXAMPLES:
        st.session_state["message_input"] = EXAMPLES[choice]
        st.session_state.pop("last_result", None)


def clear_input():
    st.session_state["message_input"] = ""
    st.session_state["example_choice"] = EXAMPLE_PLACEHOLDER
    st.session_state.pop("last_result", None)


st.session_state.setdefault("nav_page", PAGES[0])
st.session_state.setdefault("message_input", "")
st.session_state.setdefault("history", [])

# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    if os.path.exists(os.path.join(ASSETS_DIR, "logo.png")):
        st.image(os.path.join(ASSETS_DIR, "logo.png"), width=84)
    st.markdown(f"### {APP_NAME}")
    st.caption(APP_TAGLINE)
    page = st.radio("Navigation", PAGES, key="nav_page", label_visibility="collapsed")
    st.markdown("---")
    if is_model_available():
        _meta = get_metadata()
        st.markdown(
            card("Model online", _meta.get("best_model", "Unknown"),
                 f"F1 {_meta.get('metrics', {}).get('f1_score', 0):.1%} · v{_meta.get('app_version', '1.0.0')}",
                 accent="#2ecc71"),
            unsafe_allow_html=True,
        )
    else:
        st.error("No trained model found.\nRun `python src/train_model.py`")
    st.caption("🔒 Messages are analyzed in this app. Only metadata (label, score, length) is logged, never message text.")

# ============================================================================
# HOME
# ============================================================================
if page == PAGES[0]:
    logo = _logo_b64()
    img = f'<img src="data:image/png;base64,{logo}" alt="logo"/>' if logo else ""
    st.markdown(
        f'<div class="hero">{img}<div><div class="hero-title">{APP_NAME}</div>'
        f'<div class="hero-sub">{APP_TAGLINE}</div>'
        '<div class="hero-chips"><span>TF-IDF</span><span>Linear SVM</span><span>Explainable AI</span>'
        '<span>Batch CSV analysis</span><span>Risk scoring</span></div></div></div>',
        unsafe_allow_html=True,
    )

    if is_model_available():
        meta = get_metadata()
        m = meta.get("metrics", {})
        cols = st.columns(4)
        cols[0].markdown(card("Status", "Online", "Model loaded and ready", "#2ecc71"), unsafe_allow_html=True)
        cols[1].markdown(card("Active model", meta.get("best_model", "-"), "Best F1 on held-out test data"), unsafe_allow_html=True)
        cols[2].markdown(card("F1 score", f"{m.get('f1_score', 0):.1%}", f"Precision {m.get('precision', 0):.1%} · Recall {m.get('recall', 0):.1%}"), unsafe_allow_html=True)
        cols[3].markdown(card("Training data", f"{meta.get('dataset_stats', {}).get('total_messages', 0):,}", "Labelled messages"), unsafe_allow_html=True)
    else:
        st.warning("The model has not been trained yet. Run `python src/train_model.py`.")

    st.markdown("#### Get started")
    c1, c2, c3 = st.columns(3)
    for col, title, desc, target in (
        (c1, "🔍 Check a message", "Paste any email or SMS and get a verdict, a risk score and the words that drove it.", PAGES[1]),
        (c2, "📂 Scan a whole file", "Upload a CSV, classify thousands of messages at once and download the results.", PAGES[2]),
        (c3, "🧠 Understand the AI", "See what the model learned, how text becomes features, and where it can be wrong.", PAGES[5]),
    ):
        with col:
            st.markdown(f'<div class="mg-card"><div class="mg-feature-title">{title}</div><div class="mg-feature">{desc}</div></div>', unsafe_allow_html=True)
            st.button("Open →", key=f"open_{target}", on_click=go_to, args=(target,), width="stretch")

    st.markdown("#### How it works")
    st.markdown(
        '<div class="flow"><span class="step">Message</span><span class="arrow">➜</span><span class="step">Clean &amp; tokenize</span>'
        '<span class="arrow">➜</span><span class="step">TF-IDF features</span><span class="arrow">➜</span>'
        '<span class="step">ML model</span><span class="arrow">➜</span><span class="step">Verdict + risk score</span>'
        '<span class="arrow">➜</span><span class="step">Word-level explanation</span></div>',
        unsafe_allow_html=True,
    )
    st.write("")
    st.markdown(
        "".join(f'<span class="chip">{t}</span>' for t in
                ("Python", "scikit-learn", "NLTK", "pandas", "Streamlit", "Plotly", "joblib", "Docker", "GitHub Actions")),
        unsafe_allow_html=True,
    )

# ============================================================================
# SPAM DETECTION
# ============================================================================
elif page == PAGES[1]:
    page_header("Spam Detection", "Paste an email or SMS message to analyze it in real time.")
    require_model()

    top = st.columns([3, 1])
    top[0].selectbox("Try an example", [EXAMPLE_PLACEHOLDER] + list(EXAMPLES), key="example_choice",
                     on_change=load_example, label_visibility="collapsed")
    top[1].button("🧹 Clear", on_click=clear_input, width="stretch")

    text = st.text_area("Message", key="message_input", height=170, label_visibility="collapsed",
                        placeholder="Paste your email/message here...")
    st.caption(f"{len(text):,} characters")

    if st.button("🔎 Analyze Email", type="primary", width="stretch"):
        if not text.strip():
            st.warning("Please enter a message to analyze.")
            st.session_state.pop("last_result", None)
        else:
            try:
                with st.spinner("Analyzing..."):
                    res = predict_message(text)
            except ModelNotTrainedError as exc:
                st.error(str(exc))
                st.stop()
            st.session_state["last_result"] = (res, text)
            st.session_state["history"].insert(0, {
                "Time": time.strftime("%H:%M:%S"), "Verdict": res.label,
                "Risk %": res.risk_score, "Message": (text[:70] + "...") if len(text) > 70 else text})
            st.session_state["history"] = st.session_state["history"][:20]

    if "last_result" in st.session_state:
        res, shown_text = st.session_state["last_result"]
        if res.error:
            st.warning(res.error)
        else:
            if res.is_spam:
                st.markdown(f'<div class="result-banner result-spam"><div class="result-icon">🚨</div><div>'
                            f'<div class="result-title">SPAM DETECTED</div><div class="result-sub">'
                            f'{res.confidence:.1f}% confidence · {res.risk_category}</div></div></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="result-banner result-ham"><div class="result-icon">✅</div><div>'
                            f'<div class="result-title">LEGITIMATE EMAIL</div><div class="result-sub">'
                            f'{res.confidence:.1f}% confidence · {res.risk_category}</div></div></div>', unsafe_allow_html=True)

            left, right = st.columns([1, 1])
            with left:
                render_gauge(res.risk_score)
            with right:
                st.markdown(card("Prediction", res.label, f"Model: {res.model_name}", "#e74c3c" if res.is_spam else "#2ecc71"), unsafe_allow_html=True)
                st.write("")
                st.markdown(card("Confidence", f"{res.confidence:.1f}%", "Confidence in the predicted label"), unsafe_allow_html=True)
                st.write("")
                st.markdown(card("Risk level", res.risk_category, "Low ≤30 · Moderate ≤60 · High ≤80 · Very high >80", risk_color(res.risk_score)), unsafe_allow_html=True)

            st.info(res.explanation_text)

            t1, t2, t3 = st.tabs(["🖍️ Highlighted message", "📊 Influential words", "🔢 Message statistics"])
            with t1:
                if res.explanation_supported and (res.top_spam_words or res.top_ham_words) and len(shown_text) <= 5000:
                    st.markdown(f'<div class="hl-box">{highlight_html(shown_text, res.top_spam_words, res.top_ham_words)}</div>', unsafe_allow_html=True)
                    st.markdown('<div class="legend"><mark style="background:rgba(231,76,60,.5)">toward spam</mark> '
                                '<mark style="background:rgba(46,204,113,.5)">toward legitimate</mark> '
                                '· darker = stronger influence (hover a word for its weight)</div>', unsafe_allow_html=True)
                else:
                    st.caption("No word-level highlighting is available for this message.")
            with t2:
                rows = [(w.word, w.contribution, "Toward SPAM") for w in res.top_spam_words] + \
                       [(w.word, w.contribution, "Toward HAM") for w in res.top_ham_words]
                if rows:
                    cdf = pd.DataFrame(rows, columns=["Word", "Contribution", "Direction"]).sort_values("Contribution")
                    fig = px.bar(cdf, x="Contribution", y="Word", orientation="h", color="Direction",
                                 color_discrete_map={"Toward SPAM": "#e74c3c", "Toward HAM": "#2ecc71"})
                    fig.update_layout(height=max(260, 28 * len(cdf)), margin=dict(l=10, r=10, t=10, b=10), legend_title_text="")
                    st.plotly_chart(fig, width="stretch")
                    st.caption("Contribution = TF-IDF weight × model coefficient (or Naive Bayes log-odds). Read directly from the trained model.")
                else:
                    st.caption("No influential words found.")
            with t3:
                s = res.stats
                items = [("Characters", s.char_count), ("Words", s.word_count), ("URLs", s.url_count), ("Uppercase words", s.uppercase_word_count),
                         ("Digits", s.digit_count), ("Special chars", s.special_char_count), ("Repeated punctuation", s.repeated_punctuation_count),
                         ("Urgency-style words", s.urgency_word_count)]
                for row in (items[:4], items[4:]):
                    cols = st.columns(4)
                    for col, (label, value) in zip(cols, row):
                        col.markdown(card(label, value), unsafe_allow_html=True)
                    st.write("")
                st.caption("These are supporting signals only. The ML model makes the actual classification.")

            st.caption("Based on message text only. For layered protection, combine with sender, header and attachment checks (see Security & About).")

    if st.session_state["history"]:
        with st.expander(f"🕘 Recent analyses this session ({len(st.session_state['history'])})"):
            st.dataframe(pd.DataFrame(st.session_state["history"]), width="stretch", hide_index=True)
            if st.button("Clear history"):
                st.session_state["history"] = []
                st.rerun()

# ============================================================================
# BATCH ANALYSIS
# ============================================================================
elif page == PAGES[2]:
    page_header("Batch Analysis", "Upload a CSV of messages, classify them all, and download the results.")
    require_model()

    st.download_button("⬇️ Download a sample CSV template",
                       "message\nWin a FREE prize now! Click here\nAre we still on for lunch tomorrow?\n",
                       file_name="sample_messages.csv", mime="text/csv")
    uploaded = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded is not None:
        try:
            try:
                batch_df = pd.read_csv(uploaded)
            except UnicodeDecodeError:
                uploaded.seek(0)
                batch_df = pd.read_csv(uploaded, encoding="latin-1")
        except Exception as exc:
            st.error(f"Could not read this file as a CSV: {exc}")
            st.stop()
        if batch_df.empty:
            st.warning("The uploaded CSV has no rows.")
            st.stop()

        default_col = detect_text_column(batch_df)
        cols = list(batch_df.columns)
        text_col = st.selectbox("Column containing the message text", cols, index=cols.index(default_col))
        st.caption(f"{len(batch_df):,} rows found" + (f" · only the first {MAX_BATCH_ROWS:,} will be analyzed" if len(batch_df) > MAX_BATCH_ROWS else ""))
        st.dataframe(batch_df.head(5), width="stretch", hide_index=True)

        if st.button("🔎 Analyze Batch", type="primary", width="stretch"):
            bar = st.progress(0.0, text="Classifying messages...")
            out = analyze_dataframe(batch_df, text_col,
                                    progress_callback=lambda done, total: bar.progress(done / total, text=f"Classified {done:,} of {total:,}"))
            bar.empty()
            st.session_state["batch_out"] = out
            st.toast("Batch analysis complete", icon="✅")

    out = st.session_state.get("batch_out")
    if out is not None:
        spam_n = int((out["prediction"] == "SPAM").sum())
        ham_n = int((out["prediction"] == "HAM").sum())
        skipped = len(out) - spam_n - ham_n
        c = st.columns(4)
        c[0].markdown(card("Total analyzed", f"{len(out):,}"), unsafe_allow_html=True)
        c[1].markdown(card("Flagged SPAM", f"{spam_n:,}", f"{spam_n / max(len(out), 1):.1%} of messages", "#e74c3c"), unsafe_allow_html=True)
        c[2].markdown(card("Legitimate", f"{ham_n:,}", "", "#2ecc71"), unsafe_allow_html=True)
        c[3].markdown(card("Skipped (empty)", f"{skipped:,}"), unsafe_allow_html=True)
        st.write("")

        g1, g2 = st.columns(2)
        with g1:
            fig = px.pie(names=["SPAM", "HAM", "Empty"], values=[spam_n, ham_n, skipped], hole=.55,
                         color=["SPAM", "HAM", "Empty"], color_discrete_map={"SPAM": "#e74c3c", "HAM": "#2ecc71", "Empty": "#95a5a6"})
            fig.update_layout(height=280, margin=dict(t=30, b=10), title="Verdicts")
            st.plotly_chart(fig, width="stretch")
        with g2:
            order = ["Low Risk", "Moderate Risk", "High Risk", "Very High Risk"]
            rc = out["risk_category"].value_counts().reindex(order).fillna(0).reset_index()
            rc.columns = ["Risk category", "Messages"]
            fig = px.bar(rc, x="Risk category", y="Messages", color="Risk category",
                         color_discrete_sequence=["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c"])
            fig.update_layout(height=280, margin=dict(t=30, b=10), title="Risk distribution", showlegend=False)
            st.plotly_chart(fig, width="stretch")

        st.dataframe(out, width="stretch", height=380, hide_index=True,
                     column_config={"spam_risk_score": st.column_config.ProgressColumn("spam_risk_score", min_value=0, max_value=100, format="%.1f")})
        st.download_button("⬇️ Download results CSV", out.to_csv(index=False).encode("utf-8"),
                           file_name="mailguard_results.csv", mime="text/csv", type="primary")

# ============================================================================
# MODEL PERFORMANCE
# ============================================================================
elif page == PAGES[3]:
    page_header("Model Performance", "How the candidate models compare on unseen test data.")
    require_model()
    meta = get_metadata()
    metrics_path = os.path.join(OUTPUTS_DIR, "model_metrics.csv")
    if not os.path.exists(metrics_path):
        st.warning("Metrics file not found. Re-run training to regenerate it.")
        st.stop()

    mdf = pd.read_csv(metrics_path)
    best = mdf[mdf["model"] == meta.get("best_model")].iloc[0]
    st.markdown(f"**Selected model: {meta.get('best_model')}**")
    cols = st.columns(5)
    for col, (label, key) in zip(cols, (("Accuracy", "accuracy"), ("Precision", "precision"), ("Recall", "recall"),
                                         ("F1 score", "f1_score"), ("False-positive rate", "false_positive_rate"))):
        col.markdown(card(label, f"{best[key]:.2%}", accent="#11998e"), unsafe_allow_html=True)
    st.write("")

    t1, t2, t3 = st.tabs(["⚖️ Compare models", "🧮 Confusion matrix", "🎯 How the model was chosen"])
    with t1:
        show = mdf[["model", "accuracy", "precision", "recall", "f1_score", "false_positive_rate"]]
        st.dataframe(show.style.format({c: "{:.4f}" for c in show.columns if c != "model"}), width="stretch", hide_index=True)
        long = mdf.melt(id_vars="model", value_vars=["accuracy", "precision", "recall", "f1_score"])
        fig = px.bar(long, x="model", y="value", color="variable", barmode="group", labels={"value": "Score", "variable": "Metric", "model": ""})
        fig.update_layout(yaxis_range=[0, 1.05], height=380, margin=dict(t=20))
        st.plotly_chart(fig, width="stretch")
        fig2 = px.bar(mdf, x="model", y="false_positive_rate", labels={"false_positive_rate": "False-positive rate", "model": ""},
                      title="False-positive rate (legitimate messages wrongly flagged; lower is better)", color_discrete_sequence=["#e67e22"])
        fig2.update_layout(height=300)
        st.plotly_chart(fig2, width="stretch")
    with t2:
        choice = st.selectbox("Model", list(mdf["model"]), index=list(mdf["model"]).index(meta.get("best_model")))
        r = mdf[mdf["model"] == choice].iloc[0]
        z = [[int(r["true_negative"]), int(r["false_positive"])], [int(r["false_negative"]), int(r["true_positive"])]]
        fig = px.imshow(z, text_auto=True, x=["Predicted HAM", "Predicted SPAM"], y=["Actual HAM", "Actual SPAM"],
                        color_continuous_scale="Teal", aspect="auto")
        fig.update_layout(height=380, coloraxis_showscale=False, margin=dict(t=20))
        st.plotly_chart(fig, width="stretch")
        st.caption(f"{int(r['false_positive'])} legitimate message(s) wrongly flagged as spam and {int(r['false_negative'])} spam message(s) missed on the held-out test set.")
    with t3:
        st.markdown(
            "Models are ranked by **F1 score**. If several are within one percentage point of the top F1, the one with the "
            "**higher precision** wins, because wrongly flagging a real email is usually costlier than letting one spam message through. "
            "All metrics come from a stratified 80/20 split; the TF-IDF vectorizer is fitted on the training portion only (no data leakage)."
        )

# ============================================================================
# DATASET INSIGHTS
# ============================================================================
elif page == PAGES[4]:
    page_header("Dataset Insights", "What the model was trained on.")
    try:
        df, stats = _cached_dataset()
    except DatasetError as exc:
        st.error(str(exc))
        st.stop()

    c = st.columns(4)
    c[0].markdown(card("Total messages", f"{stats.total_messages:,}", f"{stats.duplicates_removed} duplicates removed"), unsafe_allow_html=True)
    c[1].markdown(card("Spam", f"{stats.spam_count:,}", accent="#e74c3c"), unsafe_allow_html=True)
    c[2].markdown(card("Ham (legitimate)", f"{stats.ham_count:,}", accent="#2ecc71"), unsafe_allow_html=True)
    c[3].markdown(card("Spam share", f"{stats.spam_percentage}%", "The classes are imbalanced"), unsafe_allow_html=True)
    st.write("")

    cmap = {"Ham": "#2ecc71", "Spam": "#e74c3c"}
    t1, t2, t3, t4, t5 = st.tabs(["Overview", "Message length", "Frequent words", "Word clouds", "Browse data"])
    with t1:
        dist = df["label"].map({0: "Ham", 1: "Spam"}).value_counts().reset_index()
        dist.columns = ["Class", "Count"]
        fig = px.pie(dist, names="Class", values="Count", hole=.5, color="Class", color_discrete_map=cmap)
        fig.update_layout(height=340, margin=dict(t=20))
        st.plotly_chart(fig, width="stretch")
    with t2:
        ld = df.assign(length=df["message"].str.len(), Class=df["label"].map({0: "Ham", 1: "Spam"}))
        fig = px.histogram(ld, x="length", color="Class", nbins=50, barmode="overlay", range_x=[0, 320],
                           color_discrete_map=cmap, labels={"length": "Characters"})
        fig.update_layout(height=340, margin=dict(t=20))
        st.plotly_chart(fig, width="stretch")
        med = ld.groupby("Class")["length"].median()
        st.caption(f"Median length: spam {med.get('Spam', 0):.0f} characters vs ham {med.get('Ham', 0):.0f} characters.")
    with t3:
        words = _top_words(20)
        a, b = st.columns(2)
        for col, name, color in ((a, "Spam", "#e74c3c"), (b, "Ham", "#2ecc71")):
            wdf = pd.DataFrame(words[name], columns=["Word", "Count"])
            fig = px.bar(wdf, x="Count", y="Word", orientation="h", title=f"Top words in {name.lower()}", color_discrete_sequence=[color])
            fig.update_layout(yaxis=dict(autorange="reversed"), height=520, margin=dict(t=40))
            col.plotly_chart(fig, width="stretch")
    with t4:
        a, b = st.columns(2)
        for col, fname, cap in ((a, "wordcloud_spam.png", "Spam messages"), (b, "wordcloud_ham.png", "Ham messages")):
            p = os.path.join(OUTPUTS_DIR, fname)
            if os.path.exists(p):
                col.image(p, caption=cap, width="stretch")
    with t5:
        f1, f2 = st.columns([1, 2])
        kind = f1.selectbox("Class", ["All", "Spam", "Ham"])
        query = f2.text_input("Search text", placeholder="e.g. prize")
        view = df.assign(label=df["label"].map({0: "Ham", 1: "Spam"}))
        if kind != "All":
            view = view[view["label"] == kind]
        if query:
            view = view[view["message"].str.contains(query, case=False, na=False, regex=False)]
        st.caption(f"{len(view):,} matching messages")
        st.dataframe(view.head(200), width="stretch", hide_index=True)

# ============================================================================
# EXPLAINABILITY
# ============================================================================
elif page == PAGES[5]:
    page_header("Explainability", "Why the model decides what it decides.")
    require_model()
    meta = get_metadata()
    vectorizer, model = _cached_vectorizer_and_model()

    st.markdown(
        f"The active model is **{meta.get('best_model')}**. Messages become numbers with **TF-IDF**: a word scores high "
        "when it appears often in *this* message but rarely across *all* messages. That is why words like “claim” or “prize” "
        "are strong spam signals while “the” or “and” barely matter."
    )

    st.markdown("#### What the model learned overall")
    top_spam, top_ham = global_top_features(vectorizer, model, top_n=15)
    if top_spam or top_ham:
        a, b = st.columns(2)
        for col, data, title, color in ((a, top_spam, "Strongest spam indicators", "#e74c3c"), (b, top_ham, "Strongest legitimate indicators", "#2ecc71")):
            fig = px.bar(pd.DataFrame(data, columns=["Feature", "Weight"]), x="Weight", y="Feature", orientation="h",
                         title=title, color_discrete_sequence=[color])
            fig.update_layout(yaxis=dict(autorange="reversed"), height=470, margin=dict(t=40))
            col.plotly_chart(fig, width="stretch")
        st.caption("Weights are read directly from the trained model, not from a hand-written keyword list.")
    else:
        st.info("Global feature weights are not available for this model type.")

    st.markdown("#### See how a message is processed")
    demo = st.text_input("Type a message", value=EXAMPLES["Prize notification"])
    cleaned = clean_text(demo)
    st.markdown("**After cleaning:** " + ("".join(f'<span class="chip">{html.escape(t)}</span>' for t in cleaned.split()) or "_(nothing left after cleaning)_"),
                unsafe_allow_html=True)
    vec = vectorizer.transform([cleaned])
    names = vectorizer.get_feature_names_out()
    feats = sorted(((names[i], float(vec[0, i])) for i in vec.nonzero()[1]), key=lambda r: -r[1])
    if feats:
        st.dataframe(pd.DataFrame(feats, columns=["TF-IDF feature", "Weight"]), width="stretch", hide_index=True, height=min(300, 40 + 35 * len(feats)))
    else:
        st.caption("None of this message's words are in the model's vocabulary.")

    st.markdown("#### Limitations")
    st.markdown(
        "- The model reads **text only**: no sender reputation, headers, attachments or SPF/DKIM/DMARC results.\n"
        "- It was trained on SMS messages, so email-specific phishing may look different to it.\n"
        "- Messages that read like normal conversation but are really promotions can be missed.\n"
        "- Probabilities for the Linear SVM come from Platt-scaling calibration, an approximation."
    )

# ============================================================================
# SECURITY & ABOUT
# ============================================================================
elif page == PAGES[6]:
    page_header("Security & About", "Security notice, methodology and release information.")
    meta = get_metadata() if is_model_available() else {}
    c = st.columns(4)
    c[0].markdown(card("App version", meta.get("app_version", "1.0.0")), unsafe_allow_html=True)
    c[1].markdown(card("Model", meta.get("best_model", "N/A")), unsafe_allow_html=True)
    c[2].markdown(card("F1 (test set)", f"{meta.get('metrics', {}).get('f1_score', 0):.2%}" if meta else "N/A"), unsafe_allow_html=True)
    c[3].markdown(card("Training size", f"{meta.get('dataset_stats', {}).get('total_messages', 0):,}" if meta else "N/A"), unsafe_allow_html=True)
    st.write("")

    st.warning(
        "**Security notice.** This system evaluates **message text only** and is one layer of defense, not a complete email-security "
        "solution. A full stack should also evaluate sender reputation, headers, URLs, attachments, domain information, authentication "
        "(SPF/DKIM/DMARC) and behavioral signals. No spam or phishing filter, including this one, can guarantee detection of every "
        "malicious message."
    )

    t1, t2, t3 = st.tabs(["How it's built", "Dataset & limitations", "Roadmap"])
    with t1:
        st.markdown(
            "- **Preprocessing:** lowercasing, HTML/URL normalization, tokenization, stopword removal (keeping signal words such as “free”, “win”, “not”) and lemmatization, identical at training and prediction time.\n"
            "- **Features:** TF-IDF over unigrams and bigrams, fitted on the training split only.\n"
            "- **Models compared:** Multinomial Naive Bayes, Logistic Regression and a calibrated Linear SVM.\n"
            "- **Selection:** highest F1 on a held-out test set, ties within one point broken by precision.\n"
            "- **Explainability:** per-message word contributions read from the live model."
        )
    with t2:
        st.markdown(
            "Trained on the UCI **SMS Spam Collection** (Kaggle: `uciml/sms-spam-collection-dataset`), loaded from `data/spam.csv`.\n\n"
            "- SMS text differs from full email; spoofed names, malicious attachments and lookalike domains are not represented.\n"
            "- Text-only analysis; headers and attachments are not inspected.\n"
            "- Unusual or adversarial messages can be misclassified."
        )
    with t3:
        st.markdown("- Train on a labelled email corpus\n- Add sender and header signals\n- Add a transformer-based model for comparison\n- Add a user-feedback loop for retraining")

st.markdown(f'<div class="mg-footer">{APP_NAME} · Built with Python, scikit-learn, NLTK and Streamlit</div>', unsafe_allow_html=True)
