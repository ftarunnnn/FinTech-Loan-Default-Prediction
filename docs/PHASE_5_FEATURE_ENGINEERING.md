# Phase 5 — Feature Engineering & Data Splitting

## 1. Overview
In Phase 5, advanced financial domain ratios were engineered to enhance model discrimination capacity. The dataset was then split into stratified Train, Validation, and Test sets, and feature scaling was applied strictly on the Training set to eliminate data leakage.

- **Feature Engineering Script:** [`src/feature_engineering.py`](../src/feature_engineering.py)
- **Scaler Artifact:** [`models/scaler.joblib`](../models/scaler.joblib)
- **Feature Schema Artifact:** [`models/feature_cols.joblib`](../models/feature_cols.joblib)
- **Processed Tensors:** [`data/processed/processed_tensors.npz`](../data/processed/processed_tensors.npz)

---

## 2. Engineered Financial Ratios

| Feature Name | Formula / Logic | Financial Significance |
| :--- | :--- | :--- |
| **`lti`** | $\frac{\text{loan\_amount}}{\text{annual\_income} + 1}$ | Measures loan leverage relative to annual earning power |
| **`pti`** | $\frac{\text{monthly\_payment}}{\text{monthly\_income} + 1}$ | Captures monthly debt obligation burden on cash flow |
| **`revol_util_ratio`** | $\frac{\text{revol\_bal}}{\text{annual\_income} + 1}$ | Assesses revolving debt level normalized by income |
| **`interest_risk_index`** | $\frac{\text{interest\_rate} \times \text{dti}}{\text{credit\_score} + 1}$ | High interest rate & high DTI penalized by low credit score |
| **`debt_burden_index`** | $\text{dti} \times \frac{\text{loan\_amount}}{10000}$ | Quantifies combined loan size and debt-to-income impact |

---

## 3. Stratified Data Splitting & Leakage Prevention

To ensure robust evaluation and zero data leakage:
1. **Stratification:** Maintained an identical default rate (~$43.16\%$) across all partitions.
2. **Standard Scaling:** `StandardScaler` was **fit exclusively on the Training set** (`3,500` samples) and subsequently applied to transform Validation (`750` samples) and Test (`750` samples) sets.

### Dataset Partition Summary
- **Training Set (70%):** 3,500 samples (Default Rate: $43.14\%$)
- **Validation Set (15%):** 750 samples (Default Rate: $43.20\%$)
- **Testing Set (15%):** 750 samples (Default Rate: $43.20\%$)
