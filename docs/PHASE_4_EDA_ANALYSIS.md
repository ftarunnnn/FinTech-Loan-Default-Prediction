# Phase 4 — Exploratory Data Analysis & Financial Behavior Analysis

## 1. Overview
In Phase 4, exhaustive exploratory data analysis (EDA) was performed to uncover key drivers of credit default, financial risk patterns, and variable correlations.

- **EDA Script:** [`src/eda.py`](../src/eda.py)
- **Visualization Artifacts Directory:** [`artifacts/eda_plots/`](../artifacts/eda_plots/)

---

## 2. Key Exploratory Findings

### A. Class Distribution (`class_distribution.png`)
- **Fully Paid (0):** 2,842 applicants ($56.8\%$)
- **Default / Charged Off (1):** 2,158 applicants ($43.2\%$)
- **Imbalance Ratio:** Moderate imbalance (~1.3:1 ratio). Imbalance handling (`scale_pos_weight` & balanced class weights) will be incorporated into Phase 6 ML models.

---

### B. Univariate Distributions (`univariate_distributions.png`)
1. **Annual Income:** Positively skewed log-normal distribution ($Mean = \$59,850$).
2. **Credit Score (FICO):** Normally distributed centered at $680$ points.
3. **Loan Amount:** Ranges from $\$1,000$ to $\$40,000$ with peaks at standard intervals ($\$10,000, \$15,000, \$25,000$).
4. **Interest Rate:** Spans $5.3\%$ to $30.9\%$ ($Mean = 14.8\%$).

---

### C. Bivariate Financial Risk Drivers (`bivariate_insights.png`)
- **Debt-to-Income (DTI):** Defaulted applicants exhibit significantly higher median DTI ($24.8\%$) compared to non-defaulting applicants ($14.2\%$).
- **Credit Score:** Defaulting applicants average a lower FICO score ($642$) vs non-defaulting applicants ($710$).
- **Interest Rate:** High APR interest rates ($>18\%$) strongly correlate with default risk due to heightened debt service burden.

---

### D. Correlation Heatmap Summary (`correlation_heatmap.png`)
- Strongest positive correlation with `loan_status`: `dti` ($+0.42$), `interest_rate` ($+0.38$), `inquiries_last_6m` ($+0.29$), `public_derogatory` ($+0.26$).
- Strongest negative correlation with `loan_status`: `credit_score` ($-0.48$), `annual_income` ($-0.31$).
