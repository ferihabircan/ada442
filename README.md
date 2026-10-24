# 💸 Bank Deposit AI (ADA 442)

Predicts whether a bank customer will subscribe to a term deposit, based on the UCI Bank Marketing dataset. The trained model is served as an interactive Streamlit app.

## What's inside

| File | What it does |
|---|---|
| `adaproject.ipynb` | Data exploration, preprocessing and model training. It compares Logistic Regression, KNN, SVC and Random Forest, tunes them with `GridSearchCV`, and evaluates with F1 and ROC-AUC. |
| `bank-additional.csv` | The UCI Bank Marketing data used for training. |
| `bank_model_pipeline.pkl` | The final scikit-learn pipeline, with a tuned decision threshold. |
| `bank_model_assets.pkl` | Supporting assets saved from the notebook. |
| `app.py` | The Streamlit app. |

## The app

Fill in the customer's details (job, marital status, education, contact history, previous campaign outcome and economic indicators), and the app returns:

- a **YES / NO** prediction using the optimized threshold
- the **subscription probability** as a confidence score and progress bar

## Run it locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The repo also includes a `.devcontainer` for GitHub Codespaces.

## Stack

Python · pandas · NumPy · scikit-learn · joblib · Streamlit
