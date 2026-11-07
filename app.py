import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, confusion_matrix, f1_score,
                             matthews_corrcoef, precision_score, recall_score,
                             roc_auc_score)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
SAMPLE_CSV = os.path.join(BASE_DIR, "test_data.csv")
TARGET = "pass_fail"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}

st.set_page_config(page_title="Student Pass/Fail Classifier", page_icon="🎓", layout="wide")


@st.cache_resource
def load_model(name):
    return joblib.load(os.path.join(MODEL_DIR, MODEL_FILES[name]))


def evaluate(pipe, features, y_true):
    """Predict with an already-trained pipeline (no retraining) and compute all six metrics."""
    pred = pipe.predict(features)
    proba = pipe.predict_proba(features)[:, 1]
    try:
        auc = roc_auc_score(y_true, proba)
    except ValueError:  # only one class present in the uploaded file
        auc = float("nan")
    metrics = {
        "Accuracy": accuracy_score(y_true, pred),
        "AUC": auc,
        "Precision": precision_score(y_true, pred, zero_division=0),
        "Recall": recall_score(y_true, pred, zero_division=0),
        "F1": f1_score(y_true, pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, pred),
    }
    return metrics, pred


# ------------------------------------------------------------------ header
st.title("🎓 Student Pass / Fail Predictor")
st.caption(
    "UCI Student Performance data (Portuguese course) - five classifiers trained offline; "
    "this app only loads the saved models and scores the test CSV you upload. "
    "Pass = final grade G3 >= 10."
)

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("1. Choose a model")
    model_name = st.selectbox("Classifier", list(MODEL_FILES.keys()))

    st.header("2. Provide test data")
    uploaded = st.file_uploader("Upload test CSV", type=["csv"])
    st.caption(f"The CSV needs all 30 input columns plus the target column `{TARGET}` (0 = fail, 1 = pass).")

    use_sample = False
    if os.path.exists(SAMPLE_CSV):
        use_sample = st.checkbox("Use the bundled test_data.csv instead", value=False)

# ------------------------------------------------------------------ load data
if uploaded is not None:
    data = pd.read_csv(uploaded)
    source = f"uploaded file: {uploaded.name}"
elif use_sample:
    data = pd.read_csv(SAMPLE_CSV)
    source = "bundled test_data.csv"
else:
    st.info("Upload a test CSV in the sidebar (or tick the bundled-file option) to begin.")
    st.stop()

model = load_model(model_name)
required = list(model.named_steps["prep"].feature_names_in_)

missing = [c for c in required + [TARGET] if c not in data.columns]
if missing:
    st.error(f"The CSV is missing required column(s): {', '.join(missing)}")
    st.stop()

if not set(data[TARGET].unique()).issubset({0, 1}):
    st.error(f"Column `{TARGET}` must contain only 0 and 1.")
    st.stop()

X_test = data[required]
y_test = data[TARGET].astype(int)

st.success(f"Loaded {len(data)} rows from {source}. Selected model: **{model_name}**")

# ------------------------------------------------------------------ metrics
metrics, pred = evaluate(model, X_test, y_test)

st.subheader(f"Evaluation metrics - {model_name}")
cols = st.columns(6)
for col, (label, value) in zip(cols, metrics.items()):
    col.metric(label, "n/a" if np.isnan(value) else f"{value:.4f}")

tab_cm, tab_report, tab_all, tab_data = st.tabs(
    ["Confusion matrix", "Classification report", "Compare all 5 models", "Data preview"]
)

with tab_cm:
    cm = confusion_matrix(y_test, pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay(cm, display_labels=["Fail (0)", "Pass (1)"]).plot(
        ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"{model_name}")
    st.pyplot(fig)
    tn, fp, fn, tp = cm.ravel()
    st.write(
        f"Correct fails: **{tn}**, correct passes: **{tp}**, "
        f"passes predicted as fail: **{fn}**, fails predicted as pass: **{fp}**."
    )

with tab_report:
    report = classification_report(
        y_test, pred, labels=[0, 1], target_names=["Fail (0)", "Pass (1)"],
        output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(report).T.round(4))

with tab_all:
    rows = []
    for name in MODEL_FILES:
        m, _ = evaluate(load_model(name), X_test, y_test)
        rows.append({"ML Model Name": name, **m})
    comparison = pd.DataFrame(rows).set_index("ML Model Name").round(4)
    st.dataframe(comparison.style.highlight_max(axis=0, color="#cfe8cf"))
    st.caption("Green = best value in each column on the data you provided.")

with tab_data:
    st.dataframe(data.head(50))
