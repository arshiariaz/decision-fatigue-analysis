# Cognitive Fatigue Analyzer

**An interactive behavioral data science application for predicting and analyzing cognitive decision fatigue.**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://cognitive-fatigue-analyzer.streamlit.app)

---

## Overview

Decision fatigue is a well-documented cognitive phenomenon in which prolonged decision-making gradually degrades an individual's ability to evaluate options accurately and efficiently. This application models that phenomenon using behavioral and physiological signals — and provides a real-time prediction engine for assessing cognitive fatigue states.

The framework is directly extensible to clinical behavioral monitoring contexts, including adaptive patient support systems, CGM-linked behavioral analysis in diabetes management, and occupational health applications.

---

## Live Demo

[Launch the app →](https://cognitive-fatigue-analyzer.streamlit.app)

---

## Key Findings from the Analysis

- **Hours awake** and **decisions made** are nearly equally the strongest predictors of fatigue (r ≈ 0.95 each)
- **Sleep deprivation** is the most powerful protective factor: less than 4h of sleep pushes average fatigue to 81.6/100; 8h+ drops it to 7.9
- **Task switching** compounds cognitive load dramatically: 20+ context switches predicts near-critical fatigue (87.9 avg)
- **Error rate inflection**: fatigue is near zero below a score of 50, then jumps 59x at the critical zone (75–100)
- **The caffeine paradox**: caffeine intake correlates positively with fatigue — a compensatory behavior signal, not a protective one

---

## Features

### Behavioral Dashboard
- Fatigue level distribution across 25,000 simulated behavioral states
- Hours-awake fatigue trajectory with zone thresholds
- Sleep deprivation impact analysis
- Task switching cost quantification
- Stress amplifier effect
- Error rate inflection point
- Feature correlation analysis
- Caffeine paradox visualization

### Real-Time Fatigue Predictor
- Input 10 behavioral parameters via interactive sliders
- Outputs fatigue score (0–100) and level (Low / Moderate / High)
- Population percentile comparison
- Radar chart: your behavioral profile vs dataset average
- 4 derived behavioral signals with delta vs population
- 3 recovery scenarios with predicted outcomes (sleep recovery, workload reduction, stress reduction)
- 5-hour forward fatigue trajectory projection
- Session history tracker

### Model Insights
- Classifier and regression performance metrics
- Feature importance visualization
- Full methodology documentation

---

## Tech Stack

| Layer | Tools |
|---|---|
| App framework | Streamlit |
| ML models | scikit-learn (Random Forest Classifier + Regressor) |
| Data processing | Pandas, NumPy |
| Visualization | Plotly |
| Model serialization | joblib |
| Deployment | Streamlit Cloud |

---

## Model Performance

| Metric | Value |
|---|---|
| Classifier Accuracy | 96.7% |
| 5-fold CV Accuracy | 96.6% ± 0.4% |
| Regression R² | 0.9979 |
| RMSE | 1.68 |
| MAE | 1.04 |

> **Note:** High model performance reflects the synthetic nature of the dataset, where the target variable was mathematically derived from the input features. Real-world behavioral data would typically yield 75–85% accuracy, but the feature relationships and analytical framework are grounded in established cognitive load research.

---

## Engineered Features

Beyond the 10 raw features, four behavioral signals were derived:

| Feature | Formula | Rationale |
|---|---|---|
| Sleep Deficit | max(0, 7 − sleep_hours) | Deviation from recommended sleep |
| Decision Density | decisions ÷ hours_awake | Cognitive demand per unit time |
| Cognitive Pressure | stress × cognitive_load | Combined mental burden |
| Fatigue Risk Index | (hours×0.3) + (deficit×0.4) + (errors×50) + (stress×0.3) | Composite risk from cognitive load literature |

---

## Run Locally

```bash
git clone https://github.com/arshiariaz/decision-fatigue-analysis.git
cd decision-fatigue-analysis
pip install -r requirements.txt
streamlit run app.py
```

Models are trained automatically on first run (~30 seconds).

---

## Project Structure

```
decision-fatigue-analysis/
├── data/
│   └── human_decision_fatigue_dataset.csv
├── models/                  # auto-generated on first run
├── .streamlit/
│   └── config.toml
├── app.py
├── train.py
├── requirements.txt
└── README.md
```

---

## Dataset

**Human Decision Fatigue Behavioral Dataset** — 25,000 simulated decision-making states with logically structured behavioral features modeling cause-and-effect relationships between cognitive demand, recovery, and fatigue.

Source: [Kaggle](https://www.kaggle.com) · License: CC0 Public Domain

---

## Author

**Arshia Riaz**
M.S. Data Science · The University of Texas at Austin
Google Cloud Professional Machine Learning Engineer (March 2026)

[LinkedIn](https://linkedin.com/in/arshia-riaz-99102a220) · [Portfolio](https://arshiariaz.github.io) · [GitHub](https://github.com/arshiariaz)

---

## Relevance to Behavioral Science

This framework mirrors the kind of behavioral signal modeling used in clinical decision support and patient monitoring systems. Key extensions include:

- **Diabetes management**: Pre/post-meal decision patterns in T1/T2 patients correlate with glucose management outcomes. This fatigue framework can identify high-risk decision windows.
- **CGM-linked behavioral monitoring**: Integrating continuous glucose monitor data with behavioral fatigue signals to predict patient adherence and self-management quality.
- **Occupational health**: Identifying high-fatigue states in clinical staff to reduce medical errors.