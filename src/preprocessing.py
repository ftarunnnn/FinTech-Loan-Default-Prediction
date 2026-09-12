import os
import pandas as pd
import numpy as np
import joblib

def preprocess_raw_data(input_path='data/raw/raw_loan_data.csv', output_dir='data/processed'):
    """
    Phase 3: Data Preprocessing
    - Loads raw loan dataset
    - Handles missing values
    - Removes duplicate records
    - Caps extreme non-physical outliers (IQR method)
    - Performs one-hot encoding for categorical attributes
    - Saves clean baseline dataset
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Reading raw data from: {input_path}")
    df = pd.read_csv(input_path)
    initial_rows = len(df)
    
    # 1. Remove Duplicate Records
    df = df.drop_duplicates()
    dedup_rows = len(df)
    print(f"[PREPROCESSING] Initial rows: {initial_rows}, After deduplication: {dedup_rows}")
    
    # 2. Handle Missing Values
    missing_before = df.isnull().sum().to_dict()
    print(f"[PREPROCESSING] Missing values before imputation: {missing_before}")
    
    # Median Imputation for numerical features
    if 'emp_length_years' in df.columns:
        median_emp = df['emp_length_years'].median()
        df['emp_length_years'] = df['emp_length_years'].fillna(median_emp)
        
    # 3. Outlier Handling (IQR Capping)
    num_cols_for_outliers = ['annual_income', 'dti', 'revol_bal', 'loan_amount', 'interest_rate']
    outlier_counts = {}
    
    for col in num_cols_for_outliers:
        if col in df.columns:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            # Count outliers
            n_outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            outlier_counts[col] = int(n_outliers)
            
            # Cap values
            df[col] = np.clip(df[col], lower_bound, upper_bound)
            
    print(f"[PREPROCESSING] Outliers capped per column: {outlier_counts}")
    
    # 4. Categorical Encoding (One-Hot Encoding)
    cat_cols = ['home_ownership', 'verification_status', 'loan_intent']
    if 'term_months' in df.columns:
        df['term_months'] = df['term_months'].astype(str)
        cat_cols.append('term_months')
        
    df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=False)
    
    # 5. Save Clean Dataset
    clean_path = os.path.join(output_dir, 'clean_loan_data.csv')
    df_encoded.to_csv(clean_path, index=False)
    
    print(f"[SUCCESS] Clean dataset saved to: {clean_path} ({df_encoded.shape[0]} rows, {df_encoded.shape[1]} cols)")
    return df_encoded

if __name__ == '__main__':
    preprocess_raw_data()
