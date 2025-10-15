# app.py
"""
Interactive Streamlit app for step-by-step ML workflow:
- Load built-in or user CSV dataset
- Inspect data and choose target & features
- Train/test split (manual trigger)
- Scaling (optional)
- Train models (Logistic Regression, Decision Tree, Random Forest) individually
- Evaluate models (confusion matrix, classification report, ROC)
- Compare models and automatically pick best by chosen metric (AUC/F1/Accuracy)
- Upload a CSV to run predictions with the chosen model
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO, BytesIO
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (confusion_matrix, classification_report,
                             roc_curve, auc, accuracy_score, f1_score,
                             roc_auc_score, precision_recall_fscore_support)
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
    cm = confusion_matrix(y_true, y_pred)
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
    # return median for each column
    return X.median().to_dict()

def evaluate_model(model, X_test, y_test, scaler=None, model_name="Model"):
    if scaler is not None:
        X_eval = scaler.transform(X_test)
    else:
        X_eval = X_test.values if isinstance(X_test, pd.DataFrame) else X_test
    y_pred = model.predict(X_eval)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_eval)[:, 1]
    else:
        # fallback: use decision_function or predicted labels
        try:
            y_prob = model.decision_function(X_eval)
            # if multi-class decision_function returns array; convert to probability-like via min-max
            if y_prob.ndim > 1:
                y_prob = (y_prob[:, 1] - y_prob.min()) / (y_prob.max() - y_prob.min() + 1e-12)
        except Exception:
            y_prob = (y_pred).astype(float)
    report = safe_classification_report(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    try:
        auc_score = roc_auc_score(y_test, y_prob)
    except Exception:
        auc_score = None
    return dict(
        y_pred=y_pred, y_prob=y_prob, report=report,
        accuracy=acc, f1=f1, auc=auc_score
    )

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
# Layout: Left sidebar for controls
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
        st.success("Built-in dataset loaded.")
else:
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
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

# Options for splitting, scaling, and random seed
test_size = st.sidebar.slider("Test size", 0.1, 0.5, 0.3, step=0.05)
random_state = st.sidebar.number_input("Random seed", value=42, step=1)
use_scaler = st.sidebar.checkbox("Use StandardScaler (recommended for LR)", value=True)

st.sidebar.markdown("---")
st.sidebar.write("Model selection (pick which to train)")
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
# Main area: show data & steps
# ---------------------------
st.title("Step-by-step ML pipeline — Breast Cancer (or your CSV)")

# Section: Dataset preview & target selection
st.header("1) Dataset preview & select target")
if st.session_state.df is not None:
    st.write("Preview (first 5 rows):")
    st.dataframe(st.session_state.df.head())
    # Target selection
    all_cols = st.session_state.df.columns.tolist()
    target_col = st.selectbox("Select target column (label)", options=all_cols, index=len(all_cols)-1)
    # Feature selection - default all except target
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

# Section: Train-test split
st.header("2) Train / Test split")
if st.session_state.X is None or st.session_state.y is None:
    st.warning("Please set dataset & target before splitting.")
else:
    st.write("Dataset shape:", st.session_state.X.shape)
    if st.button("Perform train/test split (stratified if possible)"):
        try:
            # Determine if y is categorical or continuous - prefer stratify if discrete
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

if st.session_state.split_done:
    st.write("Train shape:", st.session_state.X_train.shape, "Test shape:", st.session_state.X_test.shape)
    if st.checkbox("Show class distribution in train/test"):
        st.write("Train distribution:", np.bincount(st.session_state.y_train))
        st.write("Test distribution:", np.bincount(st.session_state.y_test))

# Section: Scaling
st.header("3) Scaling (optional)")
if st.session_state.split_done:
    if use_scaler and st.button("Fit StandardScaler on training set"):
        scaler = StandardScaler().fit(st.session_state.X_train)
        st.session_state.scaler = scaler
        st.success("Scaler fitted and stored in session.")
    if st.session_state.scaler is not None:
        st.write("Scaler mean (first 5):", st.session_state.scaler.mean_[:5])
        if st.button("Clear scaler"):
            st.session_state.scaler = None
            st.success("Scaler cleared.")

# Section: Train models (manual)
st.header("4) Train models (click per-model or train all)")
if not st.session_state.split_done:
    st.warning("Please complete train/test split first.")
else:
    col1, col2, col3 = st.columns(3)
    with col1:
        if train_lr and st.button("Train Logistic Regression"):
            scaler_local = st.session_state.scaler
            # Prepare X
            if scaler_local is not None:
                Xtr = scaler_local.transform(st.session_state.X_train)
            else:
                Xtr = st.session_state.X_train.values
            lr = LogisticRegression(max_iter=1000, random_state=random_state)
            lr.fit(Xtr, st.session_state.y_train)
            st.session_state.models['Logistic Regression'] = (lr, scaler_local)
            st.session_state.train_done = False
            st.success("Trained Logistic Regression.")
    with col2:
        if train_dt and st.button("Train Decision Tree"):
            Xtr = st.session_state.X_train.values
            dt = DecisionTreeClassifier(random_state=random_state)
            dt.fit(Xtr, st.session_state.y_train)
            st.session_state.models['Decision Tree'] = (dt, None)
            st.session_state.train_done = False
            st.success("Trained Decision Tree.")
    with col3:
        if train_rf and st.button("Train Random Forest"):
            Xtr = st.session_state.X_train.values
            rf = RandomForestClassifier(n_estimators=200, random_state=random_state)
            rf.fit(Xtr, st.session_state.y_train)
            st.session_state.models['Random Forest'] = (rf, None)
            st.session_state.train_done = False
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
        st.session_state.train_done = False
        st.success(f"Trained: {', '.join(trained)}")

# Section: Evaluate single model
st.header("5) Evaluate trained models (click model)")
if len(st.session_state.models) == 0:
    st.info("No trained models found. Train models in section 4.")
else:
    selected = st.selectbox("Select model to evaluate", options=list(st.session_state.models.keys()))
    if st.button("Evaluate selected model"):
        model, scaler_model = st.session_state.models[selected]
        metrics = evaluate_model(model, st.session_state.X_test, st.session_state.y_test, scaler=scaler_model, model_name=selected)
        # Save result
        st.session_state.results[selected] = metrics
        # Show classification report
        st.subheader(f"{selected} - Classification Report")
        rep = metrics['report']
        if rep:
            st.dataframe(pd.DataFrame(rep).transpose())
        else:
            st.write("Classification report unavailable.")
        # Show confusion matrix
        st.subheader("Confusion Matrix")
        fig = plt.figure(figsize=(5,4))
        plot_confusion_matrix(st.session_state.y_test, metrics['y_pred'], labels=np.unique(st.session_state.y))
        st.pyplot(fig)
        # ROC if available
        if metrics['auc'] is not None:
            st.subheader(f"ROC Curve (AUC = {metrics['auc']:.3f})")
            fig2 = plt.figure(figsize=(5,4))
            plot_roc(st.session_state.y_test, metrics['y_prob'], ax=fig2.gca(), label=selected)
            st.pyplot(fig2)
        st.success("Evaluation complete.")

# Section: Compare all evaluated models and choose best
st.header("6) Compare evaluated models & pick the best")
if len(st.session_state.results) == 0:
    st.info("No evaluations yet. Evaluate models in section 5.")
else:
    # Build DataFrame of metrics
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

# Section: Prediction on new data / uploaded CSV
st.header("7) Predict on new data")
st.write("You can upload a CSV with the same feature columns or use manual single-record input.")

with st.expander("Upload CSV for batch prediction"):
    upload_pred = st.file_uploader("Upload CSV (for prediction)", type=["csv"], key="pred_csv")
    if upload_pred is not None:
        df_pred = pd.read_csv(upload_pred)
        st.write("Preview uploaded data:")
        st.dataframe(df_pred.head())
        if st.button("Run predictions on uploaded CSV"):
            # choose a model
            if st.session_state.best_model is None:
                st.warning("No best model selected. Choose or pick a best model first.")
            else:
                name = st.session_state.best_model
                model, scaler_model = st.session_state.models.get(name, (None, None))
                if model is None:
                    st.error("Model not found in session.")
                else:
                    # reformat df_pred columns: fill missing with training medians and reorder
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
        # To keep UI compact, we'll show the first 10 features or all if <= 10
        feature_list = st.session_state.X.columns.tolist()
        show_features = feature_list if len(feature_list) <= 10 else feature_list[:10]
        st.write(f"Showing {len(show_features)} features for manual input (you can extend code to show all).")
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
                # build full feature vector (fill other features with medians)
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

# Final notes
st.markdown("---")
st.write("""
**Notes & tips**
- Run steps in order: load dataset → select target/features → split → (optional) scale → train models → evaluate → compare/pick best → predict.
- For custom dataset, ensure feature names match between training and prediction CSVs.
- For real production use: perform cross-validation, hyper-parameter tuning, and robust validation on external datasets.
""")
