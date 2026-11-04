# Credit Risk Prediction App

A Streamlit app that predicts whether a credit card applicant is risky (target = 1) and explains which factors drive each prediction.

## Features

- **Single prediction:** fill in an applicant's profile, work/income and asset details to get a risk probability, a gauge, and the top factors raising or lowering risk.
- **Batch prediction:** upload a CSV (a template is downloadable in the app) to score many applicants, view risk by segment and download the results. If the file has a `target` column, ROC-AUC, precision, recall, accuracy and a confusion matrix are shown.
- **Model interpretation:** local feature contributions (occlusion method), global permutation importance and partial dependence. All are model-agnostic and only need `predict_proba`.
- **Adjustable threshold:** the data is highly imbalanced (about 1.7% positives), so the classification threshold (default 0.10) can be changed in the sidebar.

## How it works

The app sends applicant records to a prediction API ([ml_deployment34](https://github.com/naufalarkan2004-spec/ml_deployment34), FastAPI on Vercel) that serves a random forest model (`model_rf_v2.sav`), then renders the probabilities and explanations.

## Project files

| File | Purpose |
|---|---|
| `streamlit_app.py` | The Streamlit application |
| `data_clean.csv` | Cleaned reference data (widget options and interpretation baseline) |
| `model_rf_v2.sav` | Trained random forest model |
| `requirements.txt` | Python dependencies |

## Run locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Tech stack

Python, Streamlit, pandas, NumPy, scikit-learn, Plotly. Backend: FastAPI.
