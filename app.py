# app.py
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay, classification_report,
    roc_curve, auc, roc_auc_score, accuracy_score, f1_score
)
import joblib
from io import StringIO

sns.set_style("whitegrid")

st.set_page_config(page_title="Breast Cancer Classifier", layout="wide")

# -------------------------
# Helpers and cached funcs
# -------------------------
@st.cache_data
def load_dataset():
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")
    return X, y, data

@st.cache_resource
def train_models(X_train, X_test, y_train, y_test):
    # Standard scaler for logistic regression
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {}
    reports = {}
    probs = {}

    # Logistic Regression (use scaled data)
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_scaled, y_train)
    y_pred_lr = lr.predict(X_test_scaled)
    y_prob_lr = lr.predict_proba(X_test_scaled)[:, 1]
    models['Logistic Regression'] = (lr, scaler, True)  # model, scaler, uses_scaler

    reports['Logistic Regression'] = classification_report(y_test, y_pred_lr, output_dict=True)
    probs['Logistic Regression'] = (y_pred_lr, y_prob_lr)

    # Decision Tree
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train, y_train)
    y_pred_dt = dt.predict(X_test)
    y_prob_dt = dt.predict_proba(X_test)[:, 1]
    models['Decision Tree'] = (dt, None, False)

    reports['Decision Tree'] = classification_report(y_test, y_pred_dt, output_dict=True)
    probs['Decision Tree'] = (y_pred_dt, y_prob_dt)

    # Random Forest
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    y_prob_rf = rf.predict_proba(X_test)[:, 1]
    models['Random Forest'] = (rf, None, False)

    reports['Random Forest'] = classification_report(y_test, y_pred_rf, output_dict=True)
    probs['Random Forest'] = (y_pred_rf, y_prob_rf)

    # metrics summary
    summary = []
    for name in models.keys():
        yhat, yprob = probs[name]
        acc = accuracy_score(y_test, yhat)
        f1 = f1_score(y_test, yhat)
        auc_score = roc_auc_score(y_test, yprob)
        summary.append([name, acc, f1, auc_score])

    summary_df = pd.DataFrame(summary, columns=["Model", "Accuracy", "F1", "ROC-AUC"]).set_index("Model")
    return models, reports, probs, summary_df

def plot_confusion(y_true, y_pred, labels, ax=None):
    cm = confusion_matrix(y_true, y_pred)
    if ax is None:
        fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, cmap=plt.cm.Blues, values_format='d')
    return ax

def plot_roc(y_true, y_prob, ax=None, label=None):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"{label} (AUC={roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc='lower right')
    return ax

# -------------------------
# Main layout
# -------------------------
st.title("Breast Cancer Classification — Streamlit Demo")
st.markdown(
    """
    Interactive demo of classification models (Logistic Regression, Decision Tree, Random Forest).
    Shows dataset exploration, training results, ROC, confusion matrices, and a small imbalance demo.
    """
)

X, y, raw = load_dataset()
target_names = raw.target_names

col1, col2 = st.columns([2, 1])

with col1:
    st.header("Data preview")
    st.dataframe(pd.concat([X.iloc[:10,:], y.iloc[:10].reset_index(drop=True)], axis=1))

    st.markdown("**Feature summary (first 5 features)**")
    st.write(X.iloc[:, :5].describe().T)

    st.markdown("**Class distribution**")
    dist = y.value_counts().rename(index={0: target_names[0], 1: target_names[1]})
    st.bar_chart(dist)

with col2:
    st.header("Controls")
    st.write("Choose test size & seed")
    test_size = st.slider("Test size (fraction)", 0.1, 0.5, 0.3, step=0.05)
    seed = st.number_input("Random seed", value=42, step=1)
    if st.button("Train / Retrain models"):
        st.session_state['train'] = True

# Initial split and training (cached)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
models, reports, probs, summary_df = train_models(X_train, X_test, y_train, y_test)

st.subheader("Model comparison (summary)")
st.dataframe(summary_df)

# Show detailed results
model_select = st.selectbox("Select model to view details", options=list(models.keys()))
model_obj, model_scaler, model_uses_scaler = models[model_select]
yhat, yprob = probs[model_select]

st.subheader(f"{model_select} — Metrics & Plots")
# Classification report table
report = reports[model_select]
report_df = pd.DataFrame(report).transpose()
st.write(report_df)

# Confusion matrix
fig, ax = plt.subplots(figsize=(5,4))
plot_confusion(y_test, yhat, labels=target_names, ax=ax)
st.pyplot(fig)

# ROC curve
fig2, ax2 = plt.subplots(figsize=(5,4))
plot_roc(y_test, yprob, ax=ax2, label=model_select)
st.pyplot(fig2)

# Feature importance for tree models
if model_select in ["Decision Tree", "Random Forest"]:
    st.subheader("Feature importances")
    importances = model_obj.feature_importances_
    feat_imp = pd.Series(importances, index=X.columns).sort_values(ascending=False).head(10)
    st.bar_chart(feat_imp)

st.markdown("---")

# -------------------------
# Imbalanced-data demo
# -------------------------
st.header("Imbalanced Data Demonstration")
st.write(
    "We create a synthetic dataset with heavy class imbalance (e.g., 98% negative, 2% positive) "
    "to illustrate pitfalls of accuracy and show class_weight='balanced'."
)

if st.button("Run imbalance demo"):
    X_syn, y_syn = make_classification(
        n_samples=5000, n_features=10, n_informative=5,
        n_redundant=2, n_clusters_per_class=1,
        weights=[0.98, 0.02], flip_y=0, random_state=42
    )
    Xs_tr, Xs_te, ys_tr, ys_te = train_test_split(X_syn, y_syn, test_size=0.3, random_state=42, stratify=y_syn)
    rf_plain = RandomForestClassifier(n_estimators=200, random_state=42)
    rf_plain.fit(Xs_tr, ys_tr)
    y_pred_plain = rf_plain.predict(Xs_te)
    y_prob_plain = rf_plain.predict_proba(Xs_te)[:, 1]

    rf_bal = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
    rf_bal.fit(Xs_tr, ys_tr)
    y_pred_bal = rf_bal.predict(Xs_te)
    y_prob_bal = rf_bal.predict_proba(Xs_te)[:, 1]

    st.write("Class distribution (train):", np.bincount(ys_tr))
    st.write("Plain RF metrics:")
    st.write(classification_report(ys_te, y_pred_plain, output_dict=False))
    st.write("Balanced RF metrics:")
    st.write(classification_report(ys_te, y_pred_bal, output_dict=False))

    # ROC comparison plot
    fig3, ax3 = plt.subplots()
    plot_roc(ys_te, y_prob_plain, ax=ax3, label="RF plain")
    plot_roc(ys_te, y_prob_bal, ax=ax3, label="RF balanced")
    st.pyplot(fig3)

st.markdown("---")

# -------------------------
# Prediction UI
# -------------------------
st.header("Make Predictions (single-row or CSV upload)")
st.write("You can provide input using sliders (single prediction) or upload a CSV file with the exact feature columns.")

with st.expander("Single-record input (sliders)"):
    sample = {}
    # pick a few representative features for sliders to keep UI compact
    selected_features = ["mean radius", "mean texture", "mean perimeter", "mean area", "mean smoothness"]
    for feat in selected_features:
        mn, mx = float(X[feat].min()), float(X[feat].max())
        val = st.slider(feat, min_value=mn, max_value=mx, value=float(X[feat].median()))
        sample[feat] = val
    if st.button("Predict single"):
        df_single = pd.DataFrame([sample])
        # need to put all columns in right order / fill missing with medians
        df_full = X.median().to_frame().T  # baseline
        for k, v in df_single.items():
            df_full[k] = v
        # keep original feature order
        df_full = df_full[X.columns]
        # use chosen model (model_select)
        mod, sc, uses = models[model_select]
        X_input = df_full.values
        if uses:
            X_input = sc.transform(X_input)
        pred = mod.predict(X_input)[0]
        prob = mod.predict_proba(X_input)[0,1]
        st.write(f"Predicted class: **{target_names[pred]}** (prob malignant = {prob:.3f})")

with st.expander("Batch prediction via CSV"):
    st.write("Upload CSV with same feature column names as the dataset.")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is not None:
        df_in = pd.read_csv(uploaded_file)
        st.write("Preview uploaded data")
        st.dataframe(df_in.head())
        # fill missing features with training medians, reorder columns
        df_full = X.median().to_frame().T.loc[[]].copy()
        # build proper DataFrame
        df_temp = pd.DataFrame(columns=X.columns)
        for c in X.columns:
            if c in df_in.columns:
                df_temp[c] = df_in[c]
            else:
                df_temp[c] = X[c].median()
        df_temp = df_temp[X.columns]
        mod, sc, uses = models[model_select]
        X_input = df_temp.values
        if uses:
            X_input = sc.transform(X_input)
        preds = mod.predict(X_input)
        probs_arr = mod.predict_proba(X_input)[:, 1]
        out = df_in.copy()
        out["pred_label"] = [target_names[p] for p in preds]
        out["prob_malignant"] = probs_arr
        st.download_button("Download predictions", out.to_csv(index=False).encode('utf-8'), "predictions.csv")
        st.dataframe(out.head())

st.markdown("---")
st.write("App created for demonstration — models are trained on the standard Breast Cancer dataset. For production use, consider cross-validation, hyperparameter tuning, interpretability (SHAP), and validation on external datasets.")
