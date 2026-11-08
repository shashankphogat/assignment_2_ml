import streamlit as st
import pandas as pd, joblib
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef,
                             confusion_matrix, classification_report)
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="ML Classifier Demo", layout="wide")
st.title("🔬 Breast Cancer Classifier — Multi-Model Comparison")

MODEL_PATHS = {
    "Logistic Regression": "model/logistic_regression.joblib",
    "Decision Tree":       "model/decision_tree.joblib",
    "KNN":                 "model/knn.joblib",
    "Naive Bayes":         "model/naive_bayes.joblib",
    "SVM":                 "model/svm.joblib",
}

uploaded = st.file_uploader("Upload test CSV (features + 'diagnosis' target)", type="csv")
model_name = st.selectbox("Select model", list(MODEL_PATHS.keys()))

if uploaded:
    df = pd.read_csv(uploaded)
    st.subheader("Preview"); st.dataframe(df.head())

    if "diagnosis" not in df.columns:
        st.error("Target column 'diagnosis' not found in uploaded CSV.")
        st.stop()

    X = df.drop(columns=["diagnosis"]); y = df["diagnosis"]
    pipe = joblib.load(MODEL_PATHS[model_name])
    y_pred  = pipe.predict(X)
    y_proba = pipe.predict_proba(X)[:, 1]

    st.subheader("📊 Evaluation Metrics")
    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy",  f"{accuracy_score(y, y_pred):.4f}")
    c2.metric("AUC",       f"{roc_auc_score(y, y_proba):.4f}")
    c3.metric("Precision", f"{precision_score(y, y_pred):.4f}")
    c4, c5, c6 = st.columns(3)
    c4.metric("Recall",    f"{recall_score(y, y_pred):.4f}")
    c5.metric("F1",        f"{f1_score(y, y_pred):.4f}")
    c6.metric("MCC",       f"{matthews_corrcoef(y, y_pred):.4f}")

    st.subheader("🧮 Confusion Matrix")
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(confusion_matrix(y, y_pred), annot=True, fmt="d", cmap="Blues",
                ax=ax, xticklabels=["Benign","Malignant"],
                yticklabels=["Benign","Malignant"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    st.pyplot(fig)

    st.subheader("📋 Classification Report")
    st.text(classification_report(y, y_pred, target_names=["Benign","Malignant"]))
else:
    st.info("Upload `test_data.csv` from the GitHub repo to begin.")
