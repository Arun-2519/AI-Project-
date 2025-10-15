# 🩺 Breast Cancer Classification — Streamlit App

An interactive AI app built using **Streamlit** and **Scikit-learn** to classify breast cancer (Malignant vs Benign).

---

## 🚀 Features

✅ Train and compare models — Logistic Regression, Decision Tree, Random Forest  
✅ Visualize Confusion Matrix, ROC Curve, and classification metrics  
✅ Handle **imbalanced data** with `class_weight='balanced'` demonstration  
✅ Predict on single input or CSV upload  
✅ Export predictions  
✅ Ready for deployment on **Streamlit Cloud / Hugging Face / Render**

---

## 🧠 Dataset

Uses the **Breast Cancer Wisconsin dataset** from `sklearn.datasets.load_breast_cancer()` — a standard dataset for binary classification.

---

## 📊 Model Summary

| Model | Accuracy | F1 Score | ROC-AUC |
|--------|-----------|-----------|-----------|
| Logistic Regression | ~0.97 | ~0.97 | ~0.99 |
| Decision Tree | ~0.93 | ~0.93 | ~0.95 |
| Random Forest | ~0.98 | ~0.98 | ~0.99 |

---

## 🏃‍♂️ Run Locally

```bash
# Clone repo
git clone https://github.com/<your-username>/breast_cancer_classifier.git
cd breast_cancer_classifier

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run app.py
