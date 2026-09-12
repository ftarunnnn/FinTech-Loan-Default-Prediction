# Phase 2 — Data Collection

## 1. Overview
In Phase 2, historical loan applicant financial data and 12-month sequential repayment trajectories were generated and structured for model development.

- **Data Generator Script:** [`src/data_generator.py`](../src/data_generator.py)
- **Static Loan Dataset:** [`data/raw/raw_loan_data.csv`](../data/raw/raw_loan_data.csv)
- **Sequential Repayment Dataset:** [`data/raw/raw_repayment_history.csv`](../data/raw/raw_repayment_history.csv)

---

## 2. Dataset Specifications

### Static Dataset (`raw_loan_data.csv`)
- **Total Records:** 5,000 applicants
- **Features:** 18 total (17 predictor variables + 1 target variable)
- **Default Rate:** $43.16\%$ (Realistic default prevalence under weighted financial risk factors)
- **Missing Values Injected:** ~3% missing in `emp_length_years` to emulate real credit bureau data gaps.

#### Summary Statistics
| Variable | Mean / Mode | Min | Max | Description |
| :--- | :--- | :--- | :--- | :--- |
| `annual_income` | $59,850.00 | $15,000.00 | $350,000.00 | Gross annual applicant income |
| `credit_score` | 680 | 350 | 850 | FICO score |
| `dti` (%) | 18.4% | 2.0% | 65.0% | Debt-to-Income percentage |
| `loan_amount` | $13,420.00 | $1,000.00 | $40,000.00 | Requested principal loan amount |
| `interest_rate` (%) | 14.8% | 5.3% | 30.9% | APR Interest Rate |

---

### Sequential Repayment Trajectory Dataset (`raw_repayment_history.csv`)
- **Dimensions:** 5,000 applicants $\times$ 12 Monthly Status Columns (`m_1` to `m_12`)
- **Encodings:**
  - `0` = On-time payment
  - `1` = 30-day delinquency
  - `2` = 60-day delinquency
  - `3` = 90+ day default / write-off
- **Purpose:** Inputs into Phase 7 **LSTM Model** to model temporal delinquency trajectories.

---

## 3. Data Integrity & Verification
- Output paths verified in `data/raw/`.
- Isolated raw directory ensuring original data remains unmutated throughout downstream processing.
