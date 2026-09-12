import os
import sys
import time

def print_header(title):
    print("\n" + "="*80)
    print(f" {title}")
    print("="*80 + "\n")

def run_end_to_end_pipeline():
    start_time = time.time()
    print_header("FinTech Loan Default Prediction System — 10-Phase Pipeline Execution")
    
    # Add src to system path
    sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
    
    # Phase 1: Problem Definition
    print_header("Phase 1: Problem Definition & Requirement Analysis")
    print("Problem formulation loaded: Binary Default Classification (0=Paid, 1=Default)")
    print("Risk Categories: Low (<30%), Medium (30-70%), High (>70%)")
    
    # Phase 2: Data Collection
    print_header("Phase 2: Data Collection")
    from data_generator import generate_fintech_dataset
    raw_dir = os.path.join('data', 'raw')
    os.makedirs(raw_dir, exist_ok=True)
    df_static, df_repayment = generate_fintech_dataset(n_samples=5000, seed=42)
    df_static.to_csv(os.path.join(raw_dir, 'raw_loan_data.csv'), index=False)
    df_repayment.to_csv(os.path.join(raw_dir, 'raw_repayment_history.csv'), index=False)
    print(f"[SUCCESS] Dataset generated with {len(df_static)} rows. Default rate: {df_static['loan_status'].mean()*100:.2f}%")
    
    # Phase 3: Data Preprocessing
    print_header("Phase 3: Data Preprocessing")
    from preprocessing import preprocess_raw_data
    df_clean = preprocess_raw_data()
    
    # Phase 4: EDA
    print_header("Phase 4: EDA & Financial Behavior Analysis")
    from eda import run_eda
    run_eda()
    
    # Phase 5: Feature Engineering & Data Splitting
    print_header("Phase 5: Feature Engineering & Data Splitting")
    from feature_engineering import run_feature_engineering
    run_feature_engineering()
    
    # Phase 6: ML Model Development
    print_header("Phase 6: ML Model Development (Random Forest & XGBoost)")
    from train_ml import train_ml_models
    train_ml_models()
    
    # Phase 7: DL Model Development
    print_header("Phase 7: DL Model Development (PyTorch ANN & LSTM)")
    from train_dl import train_dl_models
    train_dl_models()
    
    # Phase 8: Model Comparison
    print_header("Phase 8: Model Comparison & Selection")
    from evaluate_models import run_model_comparison
    run_model_comparison()
    
    # Phase 9: Risk Scoring & Explainability
    print_header("Phase 9: Risk Scoring Engine & SHAP Explainability")
    from explainability import run_explainability
    run_explainability()
    
    elapsed = time.time() - start_time
    print_header(f"All 10 Pipeline Phases Successfully Completed in {elapsed:.2f} seconds!")
    print("Run `streamlit run app.py` to launch the interactive FinTech deployment web application!")

if __name__ == '__main__':
    run_end_to_end_pipeline()
