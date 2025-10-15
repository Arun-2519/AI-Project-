# ======= app.py =======

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score, 
                             classification_report, confusion_matrix, 
                             ConfusionMatrixDisplay, roc_curve, auc)
import matplotlib.pyplot as plt

# ======= Page config =======
st.set_page_config(
    page_title="Breast Cancer Detection with ML — Interactive Explorer",
    page_icon="🎗️",
    layout="wide"
)

st.title("Breast Cancer Detection with ML — Interactive Explorer")
st.write("Upload your dataset or use the default Breast Cancer dataset, train models, evaluate performance, and explore feature meanings.")

# ======= Feature explanations =======
feature_explanations = {
    "mean radius": "Mean of distances from center to points on the perimeter",
    "mean texture": "Standard deviation of gray-scale values (measure of contrast)",
    "mean perimeter": "Mean perimeter of the nucleus",
    "mean area": "Mean area of the nucleus",
    "mean smoothness": "Mean local variation in radius lengths (smoothness)",
    "mean compactness": "Mean (perimeter^2 / area - 1.0), lower is rounder",
    "mean concavity": "Mean severity of concave portions of the contour",
    "mean concave points": "Mean number of concave portions of the contour",
    "mean symmetry": "Mean symmetry measure",
    "mean fractal dimension": "Mean complexity of nucleus boundary (like a coastline)"
}

# ======= Sidebar: Data upload =======
st.sidebar.header("Dataset Options")
uploaded_file = st.sidebar.file_uploader("Upload CSV dataset", type=["csv"])

use_default = st.sidebar.checkbox("Use default Breast Cancer dataset", value=True)

if uploaded_file:
    df = pd.read_csv(uploaded_file)
elif use_default:
    data = load_breast_cancer()
    df = pd.DataFrame(data=data.data, columns=data.feature_names)
    df['target'] = data.target
else:
    st.warning("Please upload a CSV or select default dataset")
    st.stop()

st.subheader("Dataset Preview")
st.dataframe(df.head())

# ======= Sidebar: Target & Features =======
st.sidebar.header("Select Columns")
columns = df.columns.tolist()
target_col = st.sidebar.selectbox("Select target column", options=columns, index=len(columns)-1)
feature_cols = st.sidebar.multiselect(
    "Select features for training",
    options=[c for c in columns if c != target_col],
    default=[c for c in columns if c != target_col]
)

# ======= Sidebar: Feature explanations =======
st.sidebar.header("Feature Explanations")
feature_choice = st.sidebar.selectbox("Select a feature to see explanation", options=feature_cols)
st.sidebar.write(feature_explanations.get(feature_choice, "No explanation available"))

# ======= Sidebar: Scaling =======
st.sidebar.header("Scaling Options")
scale_data = st.sidebar.checkbox("Scale features using StandardScaler", value=True)

# ======= Prepare data =======
X = df[feature_cols].values
y = df[target_col].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

if scale_data:
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
else:
    X_train_scaled, X_test_scaled = X_train, X_test

st.subheader("Train/Test Distribution")
st.write(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
st.write("Target distribution in training set:")
st.bar_chart(pd.Series(y_train).value_counts())

# ======= Train Models =======
st.subheader("Train Models")
train_button = st.button("Train Models")

if train_button:
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42)
    }

    results = []
    st.session_state["model_results"] = {}

    for name, model in models.items():
        # Logistic Regression needs scaled data
        if name == "Logistic Regression":
            model.fit(X_train_scaled, y_train)
            yhat = model.predict(X_test_scaled)
            yprob = model.predict_proba(X_test_scaled)[:,1]
        else:
            model.fit(X_train, y_train)
            yhat = model.predict(X_test)
            yprob = model.predict_proba(X_test)[:,1]
        
        acc = accuracy_score(y_test, yhat)
        f1 = f1_score(y_test, yhat, average='weighted')
        auc_score = roc_auc_score(label_binarize(y_test, classes=np.unique(y)), yprob)
        
        results.append([name, acc, f1, auc_score])
        st.session_state["model_results"][name] = {
            "model": model, "accuracy": acc, "f1": f1, "roc_auc": auc_score,
            "yhat": yhat, "yprob": yprob
        }

    df_results = pd.DataFrame(results, columns=["Model", "Accuracy", "F1 (weighted)", "ROC-AUC"])
    st.subheader("Model Comparison")
    st.dataframe(df_results.sort_values(by="ROC-AUC", ascending=False))

    # Highlight best model automatically
    best_model_name = df_results.loc[df_results["ROC-AUC"].idxmax(), "Model"]
    st.success(f"✅ Best Model based on ROC-AUC: {best_model_name}")

# ======= Evaluate Best Model =======
st.subheader("Evaluate Models")
eval_button = st.button("Evaluate All Models")

if eval_button and "model_results" in st.session_state:
    for name, info in st.session_state["model_results"].items():
        st.write(f"### {name}")
        st.write(f"Accuracy: {info['accuracy']:.4f}, F1: {info['f1']:.4f}, ROC-AUC: {info['roc_auc']:.4f}")
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, info["yhat"])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y))
        fig, ax = plt.subplots(figsize=(5,4))
        disp.plot(ax=ax, cmap=plt.cm.Blues, values_format='d')
        st.pyplot(fig)
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, info["yprob"])
        roc_auc_val = auc(fpr, tpr)
        fig2, ax2 = plt.subplots(figsize=(5,4))
        ax2.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc_val:.3f})')
        ax2.plot([0,1],[0,1],'k--', label='Random')
        ax2.set_xlabel("False Positive Rate")
        ax2.set_ylabel("True Positive Rate")
        ax2.set_title("ROC Curve")
        ax2.legend()
        st.pyplot(fig2)

    # Show best model again
    best_model_auc = max([info["roc_auc"] for info in st.session_state["model_results"].values()])
    for name, info in st.session_state["model_results"].items():
        if info["roc_auc"] == best_model_auc:
            st.success(f"🏆 Best Model after evaluation: {name} (ROC-AUC = {best_model_auc:.3f})")
            break
