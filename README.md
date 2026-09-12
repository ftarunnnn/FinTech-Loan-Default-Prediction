# FinTech — Loan Default Prediction & Risk Assessment System

![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-Enabled-green.svg)
![SHAP](https://img.shields.io/badge/Explainability-SHAP-purple.svg)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red.svg)

An end-to-end FinTech solution for predicting loan defaults, stratifying credit risk (Low, Medium, High), generating explainable SHAP drivers, and serving interactive predictions via a Streamlit web application.

---

## 🚀 10-Phase Project Workflow

| Phase | Phase Name | Description | Status |
| :---: | :--- | :--- | :---: |
| **Phase 1** | **Problem Definition & Requirement Analysis** | Define problem statement, feature taxonomy, target variable, and risk categories | ✅ Completed |
| **Phase 2** | **Data Collection** | Historical & sequential dataset synthesis (5,000 records + 12m payment history) | ✅ Completed |
| **Phase 3** | **Data Preprocessing** | Missing value imputation, outlier handling, categorical encoding | ✅ Completed |
| **Phase 4** | **EDA & Financial Analysis** | Univariate, bivariate, correlation heatmaps, behavioral insights | ✅ Completed |
| **Phase 5** | **Feature Engineering & Data Split** | Financial ratios (DTI, LTI, PTI), stratified train/val/test splits | ✅ Completed |
| **Phase 6** | **ML Model Development** | Classical ML training: Random Forest & XGBoost with class weighting | ✅ Completed |
| **Phase 7** | **DL Model Development** | Deep Learning: PyTorch ANN and sequential 12-month PyTorch LSTM | ✅ Completed |
| **Phase 8** | **Model Comparison & Selection** | Benchmark Accuracy, Precision, Recall, F1, ROC-AUC; champion election | ✅ Completed |
| **Phase 9** | **Risk Scoring & Explainability** | Continuous risk scoring (0-100%), Low/Med/High buckets, SHAP waterfall charts | ⏳ In Progress |
| **Phase 10** | **Deployment & Monitoring** | Interactive Streamlit Web App + Automated Pipeline CLI (`run_pipeline.py`) | 🔲 Planned |

---

## 📁 Repository Architecture
```
├── data/
│   ├── raw/                  # Raw synthesized loan datasets & sequential repayment logs
│   └── processed/            # Cleaned, feature-engineered splits (train, val, test)
├── docs/                     # Detailed documentation for each of the 10 phases
├── models/                   # Saved ML (joblib) & DL (PyTorch) model artifacts
├── src/                      # Modular Python pipeline modules
│   ├── data_generator.py     # Phase 2 generator
│   ├── preprocessing.py      # Phase 3 cleaner
│   ├── eda.py                # Phase 4 analysis engine
│   ├── feature_engineering.py# Phase 5 ratio & feature pipeline
│   ├── train_ml.py           # Phase 6 Random Forest & XGBoost
│   ├── train_dl.py           # Phase 7 PyTorch ANN & LSTM
│   ├── evaluate_models.py    # Phase 8 benchmark & selection
│   └── explainability.py     # Phase 9 SHAP & Risk Engine
├── artifacts/                # EDA graphs, ROC curves, confusion matrices, SHAP plots
├── app.py                    # Phase 10 Interactive Streamlit Dashboard
├── run_pipeline.py           # One-click CLI end-to-end runner
└── requirements.txt          # Dependencies
```

---

## 📌 Phase 1 Highlights
- Documented in detail in [`docs/PHASE_1_PROBLEM_DEFINITION.md`](docs/PHASE_1_PROBLEM_DEFINITION.md).
- Target: Binary `loan_status` ($0 = \text{Fully Paid}, 1 = \text{Default}$).
- Risk Stratification:
  - 🟢 **Low Risk:** Default Prob $< 30\%$
  - 🟡 **Medium Risk:** Default Prob $30\% - 70\%$
  - 🔴 **High Risk:** Default Prob $> 70\%$