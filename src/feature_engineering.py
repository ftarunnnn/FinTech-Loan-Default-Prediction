import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def run_feature_engineering(input_path='data/processed/clean_loan_data.csv', output_dir='data/processed', model_dir='models'):
    """
    Phase 5: Feature Engineering & Data Splitting
    - Computes financial ratios: LTI, PTI, Interest Risk Index, Debt Burden Index
    - Stratified Train/Val/Test Split (70 / 15 / 15)
    - Fit StandardScaler ONLY on Training data to prevent data leakage
    - Saves feature matrices and scaler artifact
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    print(f"[FEATURE ENGINEERING] Reading cleaned dataset from: {input_path}")
    df = pd.read_csv(input_path)
    
    # 1. Compute Financial Domain Features
    # Loan-to-Income Ratio
    df['lti'] = df['loan_amount'] / (df['annual_income'] + 1.0)
    
    # Estimate Monthly Payment (Simple Amortization formula)
    r = (df['interest_rate'] / 100.0) / 12.0
    n = df['term_months_36'].apply(lambda x: 36 if x == 1 else 60)
    monthly_payment = df['loan_amount'] * (r * (1 + r)**n) / ((1 + r)**n - 1)
    
    # Payment-to-Income Ratio (PTI)
    df['pti'] = monthly_payment / ((df['annual_income'] / 12.0) + 1.0)
    
    # Credit Utilization Ratio relative to income
    df['revol_util_ratio'] = df['revol_bal'] / (df['annual_income'] + 1.0)
    
    # Interest-Risk Index
    df['interest_risk_index'] = (df['interest_rate'] * df['dti']) / (df['credit_score'] + 1.0)
    
    # Debt Burden Index
    df['debt_burden_index'] = df['dti'] * (df['loan_amount'] / 10000.0)
    
    print(f"[FEATURE ENGINEERING] Created 5 engineered financial features: lti, pti, revol_util_ratio, interest_risk_index, debt_burden_index")
    
    # 2. Separate Features and Target
    drop_cols = ['applicant_id', 'loan_status']
    feature_cols = [c for c in df.columns if c not in drop_cols]
    
    X = df[feature_cols]
    y = df['loan_status']
    
    # 3. Stratified Data Splitting (70% Train, 15% Val, 15% Test)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    
    # 15% of total = 0.15 / 0.85 = 0.17647 of train_val
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.17647, random_state=42, stratify=y_train_val
    )
    
    print(f"[SPLIT] Train set size: {X_train.shape[0]} ({y_train.mean()*100:.2f}% default)")
    print(f"[SPLIT] Val set size:   {X_val.shape[0]} ({y_val.mean()*100:.2f}% default)")
    print(f"[SPLIT] Test set size:  {X_test.shape[0]} ({y_test.mean()*100:.2f}% default)")
    
    # 4. Standard Scaling (FIT ONLY ON TRAINING DATA to prevent leakage)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # 5. Save Datasets & Scaler Artifact
    scaler_path = os.path.join(model_dir, 'scaler.joblib')
    joblib.dump(scaler, scaler_path)
    joblib.dump(feature_cols, os.path.join(model_dir, 'feature_cols.joblib'))
    
    # Save CSVs with unscaled engineered features
    train_df = X_train.copy()
    train_df['loan_status'] = y_train
    train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    
    val_df = X_val.copy()
    val_df['loan_status'] = y_val
    val_df.to_csv(os.path.join(output_dir, 'val.csv'), index=False)
    
    test_df = X_test.copy()
    test_df['loan_status'] = y_test
    test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)
    
    # Save scaled numpy matrices for DL / ML models
    np.savez_compressed(
        os.path.join(output_dir, 'processed_tensors.npz'),
        X_train=X_train_scaled, y_train=y_train.values,
        X_val=X_val_scaled, y_val=y_val.values,
        X_test=X_test_scaled, y_test=y_test.values
    )
    
    print(f"[SUCCESS] Saved splits to: {output_dir}/ (train.csv, val.csv, test.csv, processed_tensors.npz)")
    print(f"[SUCCESS] Saved scaler artifact to: {scaler_path}")

if __name__ == '__main__':
    run_feature_engineering()
