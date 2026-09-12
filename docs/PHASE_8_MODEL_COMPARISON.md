# Phase 8 — Model Comparison & Selection

## 1. Overview
In Phase 8, all trained models — **Random Forest**, **XGBoost**, **PyTorch ANN**, and **PyTorch Sequential LSTM** — were benchmarked side-by-side on the unseen Test set ($750$ applicants).

- **Evaluation Script:** [`src/evaluate_models.py`](../src/evaluate_models.py)
- **JSON Metrics Summary:** [`artifacts/model_comparison.json`](../artifacts/model_comparison.json)
- **ROC Curves Plot:** [`artifacts/evaluation_plots/roc_comparison.png`](../artifacts/evaluation_plots/roc_comparison.png)

---

## 2. Benchmark Comparison Table

| Model | Model Family | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Primary Use Case |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Random Forest** | Classical ML | 0.7107 | 0.6698 | 0.6512 | 0.6604 | 0.7797 | Baseline Feature Importance |
| **XGBoost** | Gradient Boosting | 0.7080 | 0.6615 | 0.6636 | 0.6626 | 0.7767 | Production Origination Engine |
| **PyTorch ANN** | Deep Neural Net | **0.7253** | **0.6766** | **0.6975** | **0.6869** | **0.7943** | Static Application Screening |
| **PyTorch LSTM** | Recurrent Net | **0.9760** | **0.9842** | **0.9599** | **0.9719** | **0.9943** | Live Portfolio Delinquency Monitoring |

---

## 3. Champion Model Selection Rationale
Rather than blindly picking the highest raw accuracy:
1. **Pre-Issuance Loan Origination (Static Applicants):**
   - **XGBoost** and **ANN** are selected as co-champions. ANN achieves the highest static **ROC-AUC (0.7943)** and **Recall (0.6975)**, while XGBoost offers native **SHAP tree explainability** required by financial regulatory compliance (FCRA / ECOA).
2. **Post-Issuance Credit Monitoring (Active Loans):**
   - **PyTorch LSTM** is the definitive champion (**ROC-AUC 0.9943**, **Recall 0.9599**), accurately detecting 90+ day default trajectories from early payment delays.
