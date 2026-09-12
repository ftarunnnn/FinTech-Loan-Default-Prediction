# Phase 1 — Problem Definition & Requirement Analysis

## 1. Executive Summary
In retail lending and FinTech credit decisioning, accurately predicting loan defaults is critical to mitigating credit risk, optimizing capital reserves, and maintaining sustainable interest rates. The goal of this project is to build an end-to-end FinTech Loan Default Prediction System using modern Machine Learning (Random Forest, XGBoost), Deep Learning (ANN, LSTM), and Explainable AI (SHAP) techniques.

---

## 2. Problem Formulation
- **Objective:** Predict the probability that a loan applicant will default ($y = 1$) vs fully repay ($y = 0$) their obligation.
- **Problem Type:** Supervised Binary Classification with continuous probability estimation and risk stratification.
- **Business Goal:** Maximize detection of potential defaults (**Recall**) while maintaining strong precision (**F1-Score / ROC-AUC**), avoiding financial loss from false negatives.

---

## 3. Input Feature Taxonomy

| Category | Feature Name | Data Type | Description |
| :--- | :--- | :--- | :--- |
| **Demographics** | `age` | Integer | Applicant age in years |
| **Employment** | `emp_length_years` | Float | Years of employment history |
| | `home_ownership` | Categorical | RENT, OWN, MORTGAGE, OTHER |
| | `verification_status` | Categorical | Verified, Source Verified, Not Verified |
| **Financials** | `annual_income` | Float | Self-reported annual gross income ($) |
| | `credit_score` | Integer | FICO credit score (300 - 850) |
| | `revol_bal` | Float | Total revolving credit balance ($) |
| | `total_credit_lines` | Integer | Total open and closed credit lines |
| | `open_credit_lines` | Integer | Currently active credit lines |
| | `public_derogatory` | Integer | Number of derogatory public records |
| | `inquiries_last_6m` | Integer | Number of hard credit inquiries in last 6 months |
| **Loan Details** | `loan_amount` | Float | Requested principal loan amount ($) |
| | `interest_rate` | Float | Annual percentage rate (APR %) |
| | `term_months` | Categorical | 36 months, 60 months |
| | `loan_intent` | Categorical | PERSONAL, DEBT_CONSOLIDATION, BUSINESS, MEDICAL, EDUCATION, HOME_IMPROVEMENT |
| **Behavioral** | `repayment_history_12m` | Sequence | 12-month trajectory of monthly payment status (0: On-time, 1: 30d late, 2: 60d late, 3: 90+d late) |

---

## 4. Target Variable Definition
- **`loan_status`**:
  - `0` — **Fully Paid (No Default):** Applicant repays principal + interest without severe delinquency.
  - `1` — **Charged Off (Default):** Applicant defaults or incurs 90+ day delinquency leading to write-off.

---

## 5. System Output Specifications

### A. Default Probability Score
- $P(\text{Default} \mid X) \in [0.0, 1.0]$

### B. Risk Category Stratification
```
 ┌───────────────────┬───────────────────┬────────────────┐
 │ Probability Range │ Risk Category     │ Business Action│
 ├───────────────────┼───────────────────┼────────────────┤
 │ 0.00 – 0.30       │ 🟢 Low Risk       │ Auto-Approve   │
 │ 0.30 – 0.70       │ 🟡 Medium Risk    │ Manual Review  │
 │ 0.70 – 1.00       │ 🔴 High Risk      │ Auto-Decline   │
 └───────────────────┴───────────────────┴────────────────┘
```

### C. Explainable Risk Drivers
- Individual SHAP waterfall contribution breakdown (e.g. "DTI of 42.5% increased default risk by +24%").

---

## 6. Evaluation Metric Hierarchy
1. **Recall (Sensitivity):** Primary metric. Missing a default ($FN$) costs up to $100\%$ of principal amount.
2. **F1-Score / PR-AUC:** Measures harmonic balance between Recall and Precision under class imbalance.
3. **ROC-AUC:** Overall discrimination capacity across decision thresholds.
4. **Accuracy:** Supplementary reference metric (secondary to Recall).

---

## 7. 10-Phase Project Roadmap Overview
1. **Phase 1:** Problem Definition & Requirement Analysis *(Current)*
2. **Phase 2:** Data Collection & Trajectory Synthesis
3. **Phase 3:** Data Preprocessing & Outlier Sanitation
4. **Phase 4:** EDA & Financial Behavior Analysis
5. **Phase 5:** Feature Engineering & Stratified Data Splitting
6. **Phase 6:** Classical ML Model Development (Random Forest, XGBoost)
7. **Phase 7:** Deep Learning Development (ANN, Sequential LSTM)
8. **Phase 8:** Model Comparison & Champion Selection
9. **Phase 9:** Risk Scoring Engine & SHAP Explainability
10. **Phase 10:** Streamlit Web Application Deployment & Automated Pipeline CLI
