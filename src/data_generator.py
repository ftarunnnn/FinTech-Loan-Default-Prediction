import os
import numpy as np
import pandas as pd

def generate_fintech_dataset(n_samples=5000, seed=42):
    """
    Synthesizes a realistic FinTech loan default dataset and sequential 12-month repayment trajectories.
    """
    np.random.seed(seed)
    
    # 1. Demographics & Employment
    applicant_ids = [f"LN-{10000 + i}" for i in range(n_samples)]
    age = np.random.randint(21, 68, size=n_samples)
    emp_length = np.random.exponential(scale=4.5, size=n_samples)
    emp_length = np.clip(np.round(emp_length, 1), 0.0, 25.0)
    
    home_ownership = np.random.choice(
        ['RENT', 'MORTGAGE', 'OWN', 'OTHER'],
        size=n_samples,
        p=[0.45, 0.42, 0.11, 0.02]
    )
    
    verification_status = np.random.choice(
        ['Verified', 'Source Verified', 'Not Verified'],
        size=n_samples,
        p=[0.38, 0.35, 0.27]
    )
    
    # 2. Financial Metrics
    annual_income = np.random.lognormal(mean=10.9, sigma=0.55, size=n_samples)
    annual_income = np.clip(np.round(annual_income, -2), 15000, 350000)
    
    # FICO Credit Score (300 to 850)
    credit_score = np.random.normal(loc=680, scale=50, size=n_samples)
    credit_score = np.clip(np.round(credit_score), 350, 850).astype(int)
    
    # Debt-to-Income (DTI) %
    dti = np.random.gamma(shape=3.0, scale=6.0, size=n_samples)
    dti = np.clip(np.round(dti, 2), 2.0, 65.0)
    
    # Credit Lines & Inquiries
    total_credit_lines = np.random.poisson(lam=18, size=n_samples)
    total_credit_lines = np.clip(total_credit_lines, 2, 50)
    
    open_credit_lines = np.random.binomial(n=total_credit_lines, p=0.55)
    open_credit_lines = np.clip(open_credit_lines, 1, total_credit_lines)
    
    revol_bal = np.random.exponential(scale=12000, size=n_samples)
    revol_bal = np.clip(np.round(revol_bal, 2), 0.0, 150000.0)
    
    public_derogatory = np.random.choice([0, 1, 2, 3], size=n_samples, p=[0.82, 0.12, 0.04, 0.02])
    inquiries_last_6m = np.random.choice([0, 1, 2, 3, 4, 5], size=n_samples, p=[0.50, 0.25, 0.12, 0.07, 0.04, 0.02])
    
    # 3. Loan Features
    loan_intent = np.random.choice(
        ['PERSONAL', 'DEBT_CONSOLIDATION', 'BUSINESS', 'MEDICAL', 'EDUCATION', 'HOME_IMPROVEMENT'],
        size=n_samples,
        p=[0.20, 0.45, 0.12, 0.08, 0.05, 0.10]
    )
    
    term_months = np.random.choice([36, 60], size=n_samples, p=[0.70, 0.30])
    
    # Loan amount depends somewhat on income
    max_affordable_loan = annual_income * 0.45
    requested_loan = np.random.exponential(scale=14000, size=n_samples)
    loan_amount = np.minimum(requested_loan, max_affordable_loan)
    loan_amount = np.clip(np.round(loan_amount, -2), 1000, 40000)
    
    # Interest Rate (%) based on credit score & loan amount
    base_interest = 22.0 - (credit_score - 350) * (16.0 / 500.0)
    risk_bump = (dti / 100.0) * 8.0 + (term_months / 36.0) * 1.5
    interest_rate = base_interest + risk_bump + np.random.normal(0, 1.2, size=n_samples)
    interest_rate = np.clip(np.round(interest_rate, 2), 5.3, 30.9)
    
    # 4. Generate Target (loan_status: 0 = Fully Paid, 1 = Default)
    # Log-odds probability model with non-linear financial interactions
    score_term = (700.0 - credit_score) / 60.0
    dti_term = (dti - 15.0) / 10.0
    income_term = -np.log(annual_income / 50000.0) * 0.8
    loan_income_ratio = (loan_amount / annual_income) * 3.5
    derog_term = public_derogatory * 0.6
    inquiry_term = inquiries_last_6m * 0.35
    home_term = np.where(home_ownership == 'RENT', 0.4, np.where(home_ownership == 'OWN', -0.3, 0.0))
    intent_term = np.where(loan_intent == 'BUSINESS', 0.5, np.where(loan_intent == 'DEBT_CONSOLIDATION', 0.1, -0.1))
    
    logits = -2.3 + score_term + dti_term + income_term + loan_income_ratio + derog_term + inquiry_term + home_term + intent_term
    prob_default = 1.0 / (1.0 + np.exp(-logits))
    
    loan_status = np.random.binomial(1, prob_default)
    
    # 5. Inject realistic Missing Values (~3% in emp_length and inquiries)
    emp_length_missing = np.random.rand(n_samples) < 0.03
    emp_length[emp_length_missing] = np.nan
    
    # Assemble Static DataFrame
    df_static = pd.DataFrame({
        'applicant_id': applicant_ids,
        'age': age,
        'emp_length_years': emp_length,
        'home_ownership': home_ownership,
        'verification_status': verification_status,
        'annual_income': annual_income,
        'credit_score': credit_score,
        'dti': dti,
        'revol_bal': revol_bal,
        'total_credit_lines': total_credit_lines,
        'open_credit_lines': open_credit_lines,
        'public_derogatory': public_derogatory,
        'inquiries_last_6m': inquiries_last_6m,
        'loan_amount': loan_amount,
        'interest_rate': interest_rate,
        'term_months': term_months,
        'loan_intent': loan_intent,
        'loan_status': loan_status
    })
    
    # 6. Generate 12-Month Sequential Repayment History for LSTM
    # Status codes: 0 = On time, 1 = 30d late, 2 = 60d late, 3 = 90+d late / default
    repayment_matrix = np.zeros((n_samples, 12), dtype=int)
    
    for i in range(n_samples):
        is_def = loan_status[i] == 1
        base_late_prob = 0.35 if is_def else 0.04
        current_status = 0
        for m in range(12):
            if current_status >= 3:
                repayment_matrix[i, m:] = 3
                break
            
            # Transition probability
            if np.random.rand() < base_late_prob + (m * 0.02 if is_def else 0.0):
                current_status = min(3, current_status + 1)
            else:
                current_status = max(0, current_status - 1)
            repayment_matrix[i, m] = current_status
            
    repayment_cols = [f"m_{m+1}" for m in range(12)]
    df_repayment = pd.DataFrame(repayment_matrix, columns=repayment_cols)
    df_repayment.insert(0, 'applicant_id', applicant_ids)
    df_repayment['loan_status'] = loan_status
    
    return df_static, df_repayment

if __name__ == '__main__':
    raw_dir = os.path.join('data', 'raw')
    os.makedirs(raw_dir, exist_ok=True)
    
    print("Generating FinTech synthetic datasets...")
    df_static, df_repayment = generate_fintech_dataset(n_samples=5000, seed=42)
    
    static_path = os.path.join(raw_dir, 'raw_loan_data.csv')
    repayment_path = os.path.join(raw_dir, 'raw_repayment_history.csv')
    
    df_static.to_csv(static_path, index=False)
    df_repayment.to_csv(repayment_path, index=False)
    
    print(f"[SUCCESS] Generated static dataset: {static_path} ({df_static.shape[0]} rows, {df_static.shape[1]} cols)")
    print(f"[SUCCESS] Default Rate: {df_static['loan_status'].mean()*100:.2f}%")
    print(f"[SUCCESS] Generated repayment trajectory dataset: {repayment_path} ({df_repayment.shape[0]} rows, {df_repayment.shape[1]} cols)")
