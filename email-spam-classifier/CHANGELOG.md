# Changelog

All notable changes to MailGuard AI are documented here.

## [1.1.0] -- UI/UX overhaul

### Added
- Redesigned interface: hero banner, card-based metrics, tabbed pages, light/dark-mode support.
- Smart message highlighting: influential words coloured red (spam) / green (legitimate) inside the message.
- Influential-words bar chart, example-message dropdown, clear button, session history.
- Batch analysis: progress bar, verdict and risk charts, risk-score bars, sample CSV template, 5,000-row cap.
- Interactive confusion matrix for any model, false-positive chart, dataset word-frequency charts, searchable data browser.
- "See how a message is processed" demo on the Explainability page.
- `src/batch.py` and `highlight_html()` with unit tests (28 tests total).

### Changed
- Theme follows the viewer's light/dark preference; Streamlit requirement raised to >=1.52.

## [1.0.0] -- Initial release

### Added
- Full NLP + ML pipeline: cleaning, TF-IDF, 3-model training/comparison,
  automated best-model selection, persistence.
- `src/predict.py` single-source-of-truth prediction API with confidence,
  risk scoring, and model-grounded explainability.
- Streamlit dashboard: Home, Spam Detection, Batch Analysis, Model
  Performance, Dataset Insights, Explainability, Security & About.
- Batch CSV upload/classify/download workflow.
- Custom branding: logo, favicon, themed UI (`.streamlit/config.toml`).
- 23-test pytest suite covering preprocessing, data loading, and prediction
  edge cases (empty input, very long input, batch, confidence bounds).
- Deployment artifacts: `Dockerfile`, `docker-compose.yml`, `Procfile`,
  `runtime.txt`, and a GitHub Actions CI workflow that trains the model,
  runs tests, and smoke-tests the app and Docker build on every push.
