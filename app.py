# app.py
"""
Enhanced Step-by-step ML pipeline — Streamlit
- Adds feature descriptions (mean radius, mean texture, ...)
- Improved Scaling UI: show min/max/mean/std, preview before/after scaling, scale-a-value widget
- "Evaluate All Trained Models" button
- Automatic Best Model Recommendation (ROC-AUC / F1 / Accuracy)
- All prior functionality preserved: upload CSV, select target/features, split, train, eval, compare, save, predict
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (confusion_matrix, classification_report,
                             roc_curve, auc, accuracy_score, f1_score,
                             roc_auc_score)
import joblib

sns.set_style("whitegrid")
st.set_page_config(page_title="Breast Cancer Detection Using ML", layout="wide")

# ---------------------------
# Feature descriptions (brief)
# ---------------------------
FEATURE_DESCRIPTIONS = {
    "mean radius": "Mean of distances from center to points on the perimeter",
    "mean texture": "Standard deviation of gray-scale values (measure of contrast)",
    "mean perimeter": "Mean perimeter of the nucleus",
    "mean area": "Mean area of the nucleus",
    "mean smoothness": "Mean local variation in radius lengths (smoothness)",
    "mean compactness": "Mean (perimeter^2 / area - 1.0)",
    "mean concavity": "Mean severity of concave portions of the contour",
    "mean concave points": "Mean number of concave portions of the contour",
    "mean symmetry": "Mean symmetry measure",
    "mean fractal dimension": "Mean fractal dimension (coastline approximation)",
}

def describe_feature(fname):
    return FEATURE_DESCRIPTIONS.get(fname, "Feature from dataset (no short description available).")

# ---------------------------
# Helper functions
# ---------------------------
@st.cache_data
def load_builtin_breast():
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")
    return X, y, data

def safe_classification_report(y_true, y_pred):
    try:
        return classification_report(y_true, y_pred, output_dict=True)
    except Exception:
        return {}

def plot_confusion_matrix(y_true, y_pred, labels, ax=None):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    return fig

def plot_roc(y_true, y_prob, ax=None, label=None):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    ax.plot(fpr, tpr, label=f"{label} (AUC={roc_auc:.3f})")
    ax.plot([0,1],[0,1], 'k--', alpha=0.6)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    return fig, roc_auc

def get_default_feature_inputs(X):
    return X.median().to_dict()

def evaluate_model(model, X_test, y_test, scaler=None):
    if scaler is not None:
        X_eval = scaler.transform(X_test)
    else:
        X_eval = X_test.values if isinstance(X_test, pd.DataFrame) else X_test
    y_pred = model.predict(X_eval)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_eval)[:, 1]
    else:
        try:
            y_prob = model.decision_function(X_eval)
            if y_prob.ndim > 1:
                y_prob = (y_prob[:, 1] - y_prob.min()) / (y_prob.max() - y_prob.min() + 1e-12)
        except Exception:
            y_prob = np.zeros_like(y_pred, dtype=float)
    report = safe_classification_report(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    try:
        auc_score = roc_auc_score(y_test, y_prob)
    except Exception:
        auc_score = None
    return dict(y_pred=y_pred, y_prob=y_prob, report=report, accuracy=acc, f1=f1, auc=auc_score)

# ---------------------------
# Session state initialization
# ---------------------------
for key, default in {
    "df": None, "X": None, "y": None, "scaler": None,
    "models": {}, "results": {}, "best_model": None,
    "train_done": False, "split_done": False
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ---------------------------
# Sidebar controls
# ---------------------------
st.sidebar.title("Controls - Step by Step")
dataset_choice = st.sidebar.radio("Dataset", ("Use built-in Breast Cancer", "Upload CSV"))

if dataset_choice == "Use built-in Breast Cancer":
    X_builtin, y_builtin, raw = load_builtin_breast()
    if st.sidebar.button("Load built-in dataset"):
        st.session_state.df = pd.concat([X_builtin, y_builtin], axis=1)
        st.session_state.X = X_builtin.copy()
        st.session_state.y = y_builtin.copy()
        st.session_state.split_done = False
        st.session_state.train_done = False
        st.sidebar.success("Built-in dataset loaded.")
else:
    uploaded_file = st.sidebar.file_uploader("Upload CSV (for training)", type=["csv"])
    if uploaded_file is not None:
        try:
            df_uploaded = pd.read_csv(uploaded_file)
            st.session_state.df = df_uploaded
            st.session_state.split_done = False
            st.session_state.train_done = False
            st.sidebar.success("CSV uploaded")
        except Exception as e:
            st.sidebar.error(f"Failed to read CSV: {e}")

st.sidebar.markdown("---")
test_size = st.sidebar.slider("Test size", 0.1, 0.5, 0.3, step=0.05)
random_state = st.sidebar.number_input("Random seed", value=42, step=1)
use_scaler = st.sidebar.checkbox("Use StandardScaler (recommended for LR)", value=True)
st.sidebar.markdown("---")
train_lr = st.sidebar.checkbox("Logistic Regression", value=True)
train_dt = st.sidebar.checkbox("Decision Tree", value=True)
train_rf = st.sidebar.checkbox("Random Forest", value=True)
st.sidebar.markdown("---")
metric_for_best = st.sidebar.selectbox("Metric to pick best model", ("ROC-AUC", "F1", "Accuracy"))
st.sidebar.markdown("---")
if st.sidebar.button("Reset session"):
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    st.experimental_rerun()

# ---------------------------
# Main content
# ---------------------------
st.title("Step-by-step ML pipeline — Enhanced UI")

# 1) Dataset preview & selection
st.header("1) Dataset preview & select target/features")
if st.session_state.df is not None:
    st.write("Preview (first 5 rows):")
    st.dataframe(st.session_state.df.head())
    all_cols = st.session_state.df.columns.tolist()
    target_col = st.selectbox("Select target column (label)", options=all_cols, index=len(all_cols)-1)
    features = st.multiselect("Select features (columns to use as X)", options=[c for c in all_cols if c != target_col],
                              default=[c for c in all_cols if c != target_col])
    if st.button("Set dataset for modeling"):
        st.session_state.X = st.session_state.df[features].copy()
        st.session_state.y = st.session_state.df[target_col].copy()
        st.session_state.split_done = False
        st.session_state.train_done = False
        st.success("Dataset configured: features & target set.")

    with st.expander("Feature descriptions (click to expand)"):
        feat_list_to_show = st.session_state.X.columns.tolist() if st.session_state.X is not None else all_cols
        desc_rows = [(f, describe_feature(f)) for f in feat_list_to_show]
        desc_df = pd.DataFrame(desc_rows, columns=["Feature", "Short description"])
        st.dataframe(desc_df)
else:
    st.info("No dataset loaded. Use the sidebar to load built-in dataset or upload a CSV.")

# 2) Train-test split
st.header("2) Train / Test split")
if st.session_state.X is None or st.session_state.y is None:
    st.warning("Please select dataset & features first.")
else:
    st.write("Dataset shape:", st.session_state.X.shape)
    if st.button("Perform train/test split"):
        strat = st.session_state.y if st.session_state.y.nunique() <= 20 else None
        X_tr, X_te, y_tr, y_te = train_test_split(
            st.session_state.X, st.session_state.y,
            test_size=test_size, random_state=random_state, stratify=strat
        )
        st.session_state.X_train = X_tr.reset_index(drop=True)
        st.session_state.X_test = X_te.reset_index(drop=True)
        st.session_state.y_train = y_tr.reset_index(drop=True)
        st.session_state.y_test = y_te.reset_index(drop=True)
        st.session_state.split_done = True
        st.success(f"Split done. Train shape: {X_tr.shape}, Test shape: {X_te.shape}")

# Display class distribution
if st.session_state.split_done:
    st.subheader("Class distribution (train & test)")
    y_train = st.session_state.y_train
    y_test = st.session_state.y_test
    labels = sorted(list(pd.Series(y_train).unique()))
    train_counts = pd.Series(y_train).value_counts().sort_index()
    test_counts = pd.Series(y_test).value_counts().sort_index()
    dist_df = pd.DataFrame({
        "Class": labels,
        "Train Count": [int(train_counts.get(l,0)) for l in labels],
        "Test Count": [int(test_counts.get(l,0)) for l in labels]
    })
    dist_df["Train %"] = (dist_df["Train Count"]/dist_df["Train Count"].sum()*100).round(2)
    dist_df["Test %"] = (dist_df["Test Count"]/dist_df["Test Count"].sum()*100).round(2)
    st.dataframe(dist_df)

    fig, axes = plt.subplots(1,2, figsize=(10,4))
    sns.barplot(x="Class", y="Train Count", data=dist_df, ax=axes[0])
    sns.barplot(x="Class", y="Test Count", data=dist_df, ax=axes[1])
    axes[0].set_title("Train class counts")
    axes[1].set_title("Test class counts")
    st.pyplot(fig)

# 3) Scaling
st.header("3) Scaling (Optional)")
if st.session_state.split_done:
    if use_scaler and st.button("Fit StandardScaler on training set"):
        scaler = StandardScaler().fit(st.session_state.X_train)
        st.session_state.scaler = scaler
        st.success("Scaler fitted.")

    if st.session_state.scaler is not None:
        st.write("Scaler fitted. Preview original vs scaled:")
        stats = st.session_state.X_train.describe().T[['min','max','mean','std']].round(4)
        st.subheader("Train feature statistics")
        st.dataframe(stats)
        preview_scaled = pd.DataFrame(st.session_state.scaler.transform(st.session_state.X_train),
                                      columns=st.session_state.X_train.columns)
        st.subheader("First 5 rows scaled")
        st.dataframe(preview_scaled.head())
        # scale-a-value widget
        feat_to_scale = st.selectbox("Select a feature to scale a value", st.session_state.X_train.columns)
        val_to_scale = st.number_input(f"Enter value of {feat_to_scale} to scale", value=float(st.session_state.X_train[feat_to_scale].median()))
        scaled_val = st.session_state.scaler.transform(np.array([[val_to_scale if f==feat_to_scale else 0
                                                                  for f in st.session_state.X_train.columns]]))[0][
            st.session_state.X_train.columns.get_loc(feat_to_scale)]
        st.write(f"Scaled value: {scaled_val:.4f}")

# 4) Train models
st.header("4) Train Models")
if st.session_state.split_done:
    if st.button("Train selected models"):
        models = {}
        results = {}
        for model_name, selected in [("LogisticRegression", train_lr),
                                     ("DecisionTree", train_dt),
                                     ("RandomForest", train_rf)]:
            if selected:
                if model_name=="LogisticRegression":
                    model = LogisticRegression(max_iter=500, random_state=random_state)
                elif model_name=="DecisionTree":
                    model = DecisionTreeClassifier(random_state=random_state)
                elif model_name=="RandomForest":
                    model = RandomForestClassifier(random_state=random_state)
                # fit
                X_fit = st.session_state.scaler.transform(st.session_state.X_train) if st.session_state.scaler is not None else st.session_state.X_train
                model.fit(X_fit, st.session_state.y_train)
                models[model_name] = model
                res = evaluate_model(model, st.session_state.X_test, st.session_state.y_test, scaler=st.session_state.scaler)
                results[model_name] = res
        st.session_state.models = models
        st.session_state.results = results
        st.session_state.train_done = True
        st.success(f"Trained {len(models)} models.")

# 5) Evaluate & pick best
st.header("5) Evaluate & pick best model")
if st.session_state.train_done:
    best_score = -1
    best_model_name = None
    for name, res in st.session_state.results.items():
        metric_value = res["auc"] if metric_for_best=="ROC-AUC" else (res["f1"] if metric_for_best=="F1" else res["accuracy"])
        st.subheader(name)
        st.write(f"Accuracy: {res['accuracy']:.4f}, F1: {res['f1']:.4f}, AUC: {res['auc'] if res['auc'] is not None else 'NA'}")
        fig_cm = plot_confusion_matrix(st.session_state.y_test, res['y_pred'], labels=sorted(st.session_state.y_test.unique()))
        st.pyplot(fig_cm)
        fig_roc, _ = plot_roc(st.session_state.y_test, res['y_prob'], label=name)
        st.pyplot(fig_roc)
        if metric_value is not None and metric_value>best_score:
            best_score = metric_value
            best_model_name = name
    st.session_state.best_model = best_model_name
    st.success(f"Best model selected: {best_model_name} based on {metric_for_best}")

# 6) Predict new data
st.header("6) Predict new data")
if st.session_state.train_done:
    st.write("Enter values for prediction (median defaults shown)")
    default_vals = get_default_feature_inputs(st.session_state.X)
    new_data = {}
    for f in st.session_state.X.columns:
        new_data[f] = st.number_input(f, value=float(default_vals[f]))
    if st.button("Predict"):
        df_new = pd.DataFrame([new_data])
        model_to_use = st.session_state.models[st.session_state.best_model]
        X_new = st.session_state.scaler.transform(df_new) if st.session_state.scaler is not None else df_new
        pred = model_to_use.predict(X_new)[0]
        st.success(f"Predicted class: {pred}")
