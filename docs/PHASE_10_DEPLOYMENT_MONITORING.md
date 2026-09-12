# Phase 10 — Web Application Deployment & Automated Monitoring Pipeline

## 1. Overview
Phase 10 delivers a production-style user-facing web application and an automated CLI orchestration script for end-to-end execution.

- **Web Application:** [`app.py`](../app.py) (Streamlit Interactive Web Dashboard)
- **Automated CLI Pipeline:** [`run_pipeline.py`](../run_pipeline.py)
- **Dependencies Specification:** [`requirements.txt`](../requirements.txt)

---

## 2. Interactive Web Application Architecture (`app.py`)

The Streamlit web application features a dark-mode glassmorphism theme and four dedicated functional tabs:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        FinTech Loan Default Prediction Platform                        │
├───────────────────────┬──────────────────────┬──────────────────────┬──────────────────┤
│ 🎯 Applicant Risk     │ 📁 Batch Loan        │ 📊 EDA & Financial   │ 📈 Model         │
│    Assessment         │    Evaluator         │    Analytics         │    Comparison    │
└───────────────────────┴──────────────────────┴──────────────────────┴──────────────────┘
```

### Key UI Features
1. **Inputs:** Sliders & numeric inputs for Age, Income, Employment Length, Home Ownership, Verification Status, FICO Score, DTI, Revolving Balance, Credit Lines, Inquiries, Loan Amount, Interest Rate, Loan Term, and Purpose.
2. **Backend Pipeline:** Automatically preprocesses inputs, engineers real-time financial ratios (LTI, PTI, Credit Utilization Ratio, Interest-Risk Index, Debt Burden Index), scales features, and passes them to the champion model.
3. **Outputs:**
   - **Continuous Default Probability Score** ($0.0\% - 100.0\%$).
   - **Visual Risk Badge:** `LOW RISK` (🟢), `MEDIUM RISK` (🟡), or `HIGH RISK` (🔴).
   - **Underwriting Action Recommendation** (Auto-Approve, Manual Review, Auto-Decline).
   - **SHAP Driver Breakdown:** Top 5 feature influences elevating or lowering applicant risk.

---

## 3. Automated CLI Pipeline (`run_pipeline.py`)

A single command executes all 10 phases sequentially:

```bash
python run_pipeline.py
```

### Execution Log Summary
```
================================================================================
 Phase 1: Problem Definition & Requirement Analysis
 Phase 2: Data Collection
 Phase 3: Data Preprocessing
 Phase 4: EDA & Financial Behavior Analysis
 Phase 5: Feature Engineering & Data Splitting
 Phase 6: ML Model Development (Random Forest & XGBoost)
 Phase 7: DL Model Development (PyTorch ANN & LSTM)
 Phase 8: Model Comparison & Selection
 Phase 9: Risk Scoring Engine & SHAP Explainability
 Phase 10: Web Application Deployment & Automated Pipeline CLI
================================================================================
 All 10 Pipeline Phases Successfully Completed in 27.22 seconds!
```

---

## 4. How to Launch the System locally

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run End-to-End Pipeline:**
   ```bash
   python run_pipeline.py
   ```

3. **Launch Interactive Streamlit Web Application:**
   ```bash
   streamlit run app.py
   ```
