# app.py
"""
Step-by-step ML pipeline (Streamlit)
- Load built-in Breast Cancer dataset OR upload your CSV
- Select target & features
- Train/test split (manual trigger)
- Optional scaling (StandardScaler)
- Train models individually or all at once (Logistic Regression, Decision Tree, Random Forest)
- Evaluate models: confusion matrix, classification report, ROC curve
- Compare evaluated models, pick & save best model
- Upload CSV for batch prediction or provide a single-record input
- Improved class distribution display (labels, counts, percentages + charts)
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
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
st.set_page_config(page_title="Step-by-step ML Pipeline", layout="wide")

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
    # Prepare X_test
    if scaler is not None:
        X_eval = scaler.transform(X_test)
    else:
        X_eval = X_test.values if isinstance(X_test, pd.DataFrame) else X_test
    y_pred = model.predict(X_eval)
    # probability / score
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
if "df" not in st.session_state:
    st.session_state.df = None
if "X" not in st.session_state:
    st.session_state.X = None
if "y" not in st.session_state:
    st.session_state.y = None
if "scaler" not in st.session_state:
    st.session_state.scaler = None
if "models" not in st.session_state:
    st.session_state.models = {}
if "results" not in st.session_state:
    st.session_state.results = {}
if "best_model" not in st.session_state:
    st.session_state.best_model = None
if "train_done" not in st.session_state:
    st.session_state.train_done = False
if "split_done" not in st.session_state:
    st.session_state.split_done = False

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
st.title("Step-by-step ML pipeline — Interactive")

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
        if target_col not in st.session_state.df.columns:
            st.error("Target not in dataframe")
        else:
            st.session_state.X = st.session_state.df[features].copy()
            st.session_state.y = st.session_state.df[target_col].copy()
            st.session_state.split_done = False
            st.session_state.train_done = False
            st.success("Dataset configured: features & target set.")
else:
    st.info("No dataset loaded. Use the sidebar to load built-in dataset or upload a CSV.")

# 2) Train-test split
st.header("2) Train / Test split")
if st.session_state.X is None or st.session_state.y is None:
    st.warning("Please set dataset & target before splitting.")
else:
    st.write("Dataset shape:", st.session_state.X.shape)
    if st.button("Perform train/test split (stratified if possible)"):
        try:
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
            st.session_state.train_done = False
            st.success(f"Split done. Train shape: {X_tr.shape}, Test shape: {X_te.shape}")
        except Exception as e:
            st.error(f"Split failed: {e}")

# Display class distribution (improved)
if st.session_state.split_done:
    st.subheader("Class distribution (train & test)")
    y_train = st.session_state.y_train
    y_test = st.session_state.y_test
    # determine labels (sorted unique)
    labels = sorted(list(pd.Series(y_train).unique()))
    train_counts = pd.Series(y_train).value_counts().sort_index()
    test_counts = pd.Series(y_test).value_counts().sort_index()
    # build dataframe
    dist_df = pd.DataFrame({
        "Class": labels,
        "Train Count": [int(train_counts.get(l, 0)) for l in labels],
        "Test Count": [int(test_counts.get(l, 0)) for l in labels]
    })
    dist_df["Train %"] = (dist_df["Train Count"] / dist_df["Train Count"].sum() * 100).round(2)
    dist_df["Test %"] = (dist_df["Test Count"] / dist_df["Test Count"].sum() * 100).round(2)
    st.dataframe(dist_df)

    # bar charts
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sns.barplot(x="Class", y="Train Count", data=dist_df, ax=axes[0])
    sns.barplot(x="Class", y="Test Count", data=dist_df, ax=axes[1])
    axes[0].set_title("Train class counts")
    axes[1].set_title("Test class counts")
    st.pyplot(fig)

# 3) Scaling
st.header("3) Scaling (optional)")
if st.session_state.split_done:
    if use_scaler and st.button("Fit StandardScaler on training set"):
        scaler = StandardScaler().fit(st.session_state.X_train)
        st.session_state.scaler = scaler
        st.success("Scaler fitted and stored in session.")
    if st.session_state.scaler is not None:
        st.write("Scaler mean (first 5):", np.round(st.session_state.scaler.mean_[:5], 4))
        if st.button("Clear scaler"):
            st.session_state.scaler = None
            st.success("Scaler cleared.")

# 4) Train models (manual)
st.header("4) Train models (manual or Train All)")
if not st.session_state.split_done:
    st.warning("Please complete train/test split first.")
else:
    col1, col2, col3 = st.columns(3)
    with col1:
        if train_lr and st.button("Train Logistic Regression"):
            scaler_local = st.session_state.scaler
            Xtr = scaler_local.transform(st.session_state.X_train) if scaler_local is not None else st.session_state.X_train.values
            lr = LogisticRegression(max_iter=1000, random_state=random_state)
            lr.fit(Xtr, st.session_state.y_train)
            st.session_state.models['Logistic Regression'] = (lr, scaler_local)
            st.success("Trained Logistic Regression.")
    with col2:
        if train_dt and st.button("Train Decision Tree"):
            dt = DecisionTreeClassifier(random_state=random_state)
            dt.fit(st.session_state.X_train.values, st.session_state.y_train)
            st.session_state.models['Decision Tree'] = (dt, None)
            st.success("Trained Decision Tree.")
    with col3:
        if train_rf and st.button("Train Random Forest"):
            rf = RandomForestClassifier(n_estimators=200, random_state=random_state)
            rf.fit(st.session_state.X_train.values, st.session_state.y_train)
            st.session_state.models['Random Forest'] = (rf, None)
            st.success("Trained Random Forest.")

    if st.button("Train All Selected Models"):
        trained = []
        if train_lr:
            scaler_local = st.session_state.scaler
            Xtr = scaler_local.transform(st.session_state.X_train) if scaler_local is not None else st.session_state.X_train.values
            lr = LogisticRegression(max_iter=1000, random_state=random_state)
            lr.fit(Xtr, st.session_state.y_train)
            st.session_state.models['Logistic Regression'] = (lr, scaler_local)
            trained.append("Logistic Regression")
        if train_dt:
            dt = DecisionTreeClassifier(random_state=random_state)
            dt.fit(st.session_state.X_train.values, st.session_state.y_train)
            st.session_state.models['Decision Tree'] = (dt, None)
            trained.append("Decision Tree")
        if train_rf:
            rf = RandomForestClassifier(n_estimators=200, random_state=random_state)
            rf.fit(st.session_state.X_train.values, st.session_state.y_train)
            st.session_state.models['Random Forest'] = (rf, None)
            trained.append("Random Forest")
        st.success(f"Trained: {', '.join(trained)}")

# 5) Evaluate single model
st.header("5) Evaluate trained models (select & evaluate)")
if len(st.session_state.models) == 0:
    st.info("No trained models found. Train models in section 4.")
else:
    selected = st.selectbox("Select model to evaluate", options=list(st.session_state.models.keys()))
    if st.button("Evaluate selected model"):
        model, scaler_model = st.session_state.models[selected]
        metrics = evaluate_model(model, st.session_state.X_test, st.session_state.y_test, scaler=scaler_model)
        st.session_state.results[selected] = metrics

        # show report
        st.subheader(f"{selected} - Classification Report")
        rep = metrics['report']
        if rep:
            st.dataframe(pd.DataFrame(rep).transpose())
        else:
            st.write("Classification report unavailable.")

        # confusion matrix
        st.subheader("Confusion Matrix")
        fig = plt.figure(figsize=(5,4))
        labels_sorted = sorted(list(pd.Series(st.session_state.y_test).unique()))
        plot_confusion_matrix(st.session_state.y_test, metrics['y_pred'], labels=labels_sorted, ax=fig.gca())
        st.pyplot(fig)

        # ROC
        if metrics['auc'] is not None and not np.isnan(metrics['auc']):
            st.subheader(f"ROC Curve (AUC = {metrics['auc']:.3f})")
            fig2 = plt.figure(figsize=(5,4))
            plot_roc(st.session_state.y_test, metrics['y_prob'], ax=fig2.gca(), label=selected)
            st.pyplot(fig2)
        else:
            st.write("ROC AUC not available for this model.")

# 6) Compare & pick best
st.header("6) Compare evaluated models & select best")
if len(st.session_state.results) == 0:
    st.info("No evaluations yet. Evaluate models in section 5.")
else:
    comp = []
    for name, m in st.session_state.results.items():
        comp.append({
            "Model": name,
            "Accuracy": m['accuracy'],
            "F1": m['f1'],
            "ROC-AUC": m['auc'] if m['auc'] is not None else np.nan
        })
    comp_df = pd.DataFrame(comp).set_index("Model")
    st.dataframe(comp_df)
    if st.button("Automatically pick best model"):
        sort_key = metric_for_best
        if sort_key == "ROC-AUC":
            best = comp_df['ROC-AUC'].idxmax()
        elif sort_key == "F1":
            best = comp_df['F1'].idxmax()
        else:
            best = comp_df['Accuracy'].idxmax()
        st.session_state.best_model = best
        st.success(f"Best model by {sort_key}: {best}")
    if st.session_state.best_model is not None:
        st.info(f"Currently selected best model: {st.session_state.best_model}")
        if st.button("Save best model to disk (joblib)"):
            name = st.session_state.best_model
            model, scaler_model = st.session_state.models[name]
            joblib.dump({"model": model, "scaler": scaler_model, "features": st.session_state.X.columns.tolist()}, f"best_model_{name.replace(' ','_')}.pkl")
            st.success(f"Saved best_model_{name.replace(' ','_')}.pkl")

# 7) Predict on new data
st.header("7) Predict on new data")
st.write("Upload a CSV for batch prediction, or use the Single-record input for a quick test.")

with st.expander("Upload CSV for batch prediction"):
    upload_pred = st.file_uploader("Upload CSV (for prediction)", type=["csv"], key="pred_csv")
    if upload_pred is not None:
        df_pred = pd.read_csv(upload_pred)
        st.write("Preview uploaded data:")
        st.dataframe(df_pred.head())
        if st.button("Run predictions on uploaded CSV"):
            if st.session_state.best_model is None:
                st.warning("No best model selected. Choose or pick a best model first.")
            else:
                name = st.session_state.best_model
                model, scaler_model = st.session_state.models.get(name, (None, None))
                if model is None:
                    st.error("Model not found in session.")
                else:
                    feat_cols = st.session_state.X.columns.tolist()
                    df_temp = pd.DataFrame(columns=feat_cols, index=df_pred.index)
                    for c in feat_cols:
                        if c in df_pred.columns:
                            df_temp[c] = df_pred[c]
                        else:
                            df_temp[c] = st.session_state.X[c].median()
                    X_input = df_temp.values
                    if scaler_model is not None:
                        X_input = scaler_model.transform(X_input)
                    preds = model.predict(X_input)
                    probs = model.predict_proba(X_input)[:, 1] if hasattr(model, "predict_proba") else None
                    out = df_pred.copy()
                    out["predicted_label"] = preds
                    out["pred_prob"] = probs
                    st.dataframe(out.head())
                    csv_bytes = out.to_csv(index=False).encode("utf-8")
                    st.download_button("Download predictions CSV", csv_bytes, "predictions.csv")

with st.expander("Single-record manual input"):
    if st.session_state.X is not None:
        default_inputs = get_default_feature_inputs(st.session_state.X)
        user_input = {}
        feature_list = st.session_state.X.columns.tolist()
        # show first 10 features to keep UI compact; can be extended
        show_features = feature_list if len(feature_list) <= 10 else feature_list[:10]
        st.write(f"Showing {len(show_features)} features for manual input (first {len(show_features)} features).")
        for feat in show_features:
            mn = float(st.session_state.X[feat].min())
            mx = float(st.session_state.X[feat].max())
            val = st.number_input(feat, value=float(default_inputs[feat]), min_value=mn, max_value=mx, key=f"inp_{feat}")
            user_input[feat] = val
        if st.button("Predict single record"):
            if st.session_state.best_model is None:
                st.warning("No best model selected - cannot predict.")
            else:
                name = st.session_state.best_model
                model, scaler_model = st.session_state.models[name]
                full = pd.DataFrame([st.session_state.X.median().to_dict()])
                for k, v in user_input.items():
                    full[k] = v
                full = full[st.session_state.X.columns]
                X_input = full.values
                if scaler_model is not None:
                    X_input = scaler_model.transform(X_input)
                pred = model.predict(X_input)[0]
                prob = model.predict_proba(X_input)[0, 1] if hasattr(model, "predict_proba") else None
                st.success(f"Prediction: {pred} (prob positive={prob:.3f})")

st.markdown("---")
st.write("""
**Notes**
- Run steps in order: load dataset → select target/features → split → (optional) scale → train models → evaluate → compare/pick best → predict.
- For custom dataset: ensure feature names align between training and prediction CSVs.
- For production: add cross-validation, hyperparameter tuning, and interpretability (SHAP).
""")
