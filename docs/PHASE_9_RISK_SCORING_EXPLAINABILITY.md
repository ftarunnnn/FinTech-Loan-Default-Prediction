# Phase 9 — Risk Scoring Engine & SHAP Explainability

## 1. Overview
In Phase 9, continuous predicted default probabilities $P(\text{Default})$ were mapped into risk categories (Low, Medium, High). A SHAP (SHapley Additive exPlanations) explainability engine was integrated to output global feature importance charts and local risk driver breakdowns for individual applicants.

- **Explainability Script:** [`src/explainability.py`](../src/explainability.py)
- **SHAP Global Summary Plot:** [`artifacts/shap_summary.png`](../artifacts/shap_summary.png)

---

## 2. Risk Stratification Buckets

```
 ┌───────────────────────┬───────────────┬─────────────────┬──────────────────────┐
 │ Continuous Probability│ Risk Category │ Visual Badge    │ Action Workflow      │
 ├───────────────────────┼───────────────┼─────────────────┼──────────────────────┤
 │ 0.0% – 29.9%          │ LOW           │ 🟢 Green Badge  │ Automated Approval   │
 │ 30.0% – 69.9%         │ MEDIUM        │ 🟡 Amber Badge  │ Underwriter Review   │
 │ 70.0% – 100.0%        │ HIGH          │ 🔴 Red Badge    │ Automated Decline    │
 └───────────────────────┴───────────────┴─────────────────┴──────────────────────┘
```

---

## 3. SHAP Explainability Architecture

### A. Global Feature Importance (`shap_summary.png`)
SHAP TreeExplainer evaluates feature impact direction across the test population:
1. **Top Risk-Increasing Features:** High `dti`, high `interest_risk_index`, frequent `inquiries_last_6m`, and `public_derogatory` records.
2. **Top Risk-Reducing Features:** High `credit_score`, higher `annual_income`, and favorable `lti` ratios.

### B. Local Applicant Explanation Example
For a sample applicant evaluated with $49.99\%$ default probability (**Medium Risk — Manual Review**):

```json
{
    "default_probability_pct": 49.99,
    "risk_category": "MEDIUM",
    "action": "Manual Review",
    "top_drivers": [
        {
            "feature": "inquiries_last_6m",
            "shap_impact": +0.7858,
            "impact_direction": "Elevates Risk"
        },
        {
            "feature": "lti (Loan-to-Income)",
            "shap_impact": -0.4688,
            "impact_direction": "Lowers Risk"
        },
        {
            "feature": "loan_intent_BUSINESS",
            "shap_impact": +0.3982,
            "impact_direction": "Elevates Risk"
        }
    ]
}
```
