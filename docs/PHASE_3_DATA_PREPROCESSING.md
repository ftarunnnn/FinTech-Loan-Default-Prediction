# Phase 3 — Data Preprocessing

## 1. Overview
In Phase 3, the raw static loan applicant dataset was cleaned, sanitized, and transformed into a modeling-ready baseline dataset.

- **Preprocessing Script:** [`src/preprocessing.py`](../src/preprocessing.py)
- **Output Clean Dataset:** [`data/processed/clean_loan_data.csv`](../data/processed/clean_loan_data.csv)

---

## 2. Preprocessing Steps & Audit Log

### A. Duplicate Record Removal
- Initial Rows: `5,000`
- Duplicates Found: `0`
- Post-Deduplication Rows: `5,000`

### B. Missing Value Imputation
- Feature Imputed: `emp_length_years` (`140` missing values, ~2.8%)
- Imputation Method: Median Imputation ($4.3$ years)
- Post-Imputation Missing Count: `0`

### C. Outlier Detection & IQR Capping
Extreme non-physical tail values were capped using Interquartile Range boundaries ($[Q1 - 1.5 \times IQR, Q3 + 1.5 \times IQR]$):

| Column | Outliers Detected | Action Taken |
| :--- | :--- | :--- |
| `annual_income` | 209 | Capped at upper IQR boundary |
| `dti` | 135 | Capped at upper IQR boundary |
| `revol_bal` | 234 | Capped at upper IQR boundary |
| `loan_amount` | 109 | Capped at upper IQR boundary |
| `interest_rate` | 51 | Capped at upper IQR boundary |

### D. Categorical Encoding (One-Hot)
Transformed nominal categories into binary indicator columns:
- `home_ownership` $\to$ `home_ownership_RENT`, `home_ownership_MORTGAGE`, `home_ownership_OWN`, `home_ownership_OTHER`
- `verification_status` $\to$ `verification_status_Verified`, `verification_status_Source Verified`, `verification_status_Not Verified`
- `loan_intent` $\to$ `loan_intent_PERSONAL`, `loan_intent_DEBT_CONSOLIDATION`, `loan_intent_BUSINESS`, `loan_intent_MEDICAL`, `loan_intent_EDUCATION`, `loan_intent_HOME_IMPROVEMENT`
- `term_months` $\to$ `term_months_36`, `term_months_60`

---

## 3. Dataset Dimensionality Evolution
- **Raw Features:** 18 columns
- **Processed Clean Features:** 29 columns (Post One-Hot Encoding)
- **Clean File Location:** `data/processed/clean_loan_data.csv`
