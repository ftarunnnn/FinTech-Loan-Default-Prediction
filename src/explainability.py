import os
import json
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

plt.style.use('ggplot')

def get_risk_category(prob):
    """
    Categorizes predicted default probability into financial risk buckets.
    """
    prob_pct = prob * 100.0
    if prob_pct < 30.0:
        return {'category': 'LOW', 'color': '#2ecc71', 'action': 'Auto-Approve', 'description': 'Low credit default risk'}
    elif prob_pct < 70.0:
        return {'category': 'MEDIUM', 'color': '#f39c12', 'action': 'Manual Review', 'description': 'Moderate credit default risk'}
    else:
        return {'category': 'HIGH', 'color': '#e74c3c', 'action': 'Auto-Decline', 'description': 'Elevated credit default risk'}

def run_explainability(
    model_path='models/xgboost.joblib',
    feature_cols_path='models/feature_cols.joblib',
    data_path='data/processed/processed_tensors.npz',
    output_dir='artifacts'
):
    """
    Phase 9: Risk Scoring & Explainability
    Computes SHAP values, generates global feature importance charts,
    and provides human-readable risk reason explanations for individual applicants.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"[EXPLAINABILITY] Loading XGBoost model from: {model_path}")
    
    xgb_model = joblib.load(model_path)
    feature_cols = joblib.load(feature_cols_path)
    
    data = np.load(data_path)
    X_test = data['X_test']
    
    df_test = pd.DataFrame(X_test, columns=feature_cols)
    
    # 1. Compute SHAP Values using TreeExplainer
    print("[EXPLAINABILITY] Computing SHAP values...")
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer(df_test)
    
    # 2. Save Global SHAP Feature Importance Summary Plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, df_test, show=False)
    plt.title('SHAP Global Feature Importance', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    shap_plot_path = os.path.join(output_dir, 'shap_summary.png')
    plt.savefig(shap_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SUCCESS] Saved SHAP summary plot to: {shap_plot_path}")
    
    # 3. Individual Applicant Explanation Generator
    def explain_applicant(applicant_row_idx=0):
        row_features = df_test.iloc[[applicant_row_idx]]
        prob = xgb_model.predict_proba(row_features)[0, 1]
        risk_info = get_risk_category(prob)
        
        row_shap = shap_values[applicant_row_idx].values
        top_indices = np.argsort(np.abs(row_shap))[::-1][:5]
        
        top_drivers = []
        for idx in top_indices:
            feat_name = feature_cols[idx]
            val = row_features.iloc[0, idx]
            shap_val = row_shap[idx]
            impact = "Elevates Risk" if shap_val > 0 else "Lowers Risk"
            top_drivers.append({
                'feature': feat_name,
                'value': float(round(val, 2)),
                'shap_impact': float(round(shap_val, 4)),
                'impact_direction': impact
            })
            
        return {
            'applicant_idx': applicant_row_idx,
            'default_probability_pct': float(round(prob * 100, 2)),
            'risk_category': risk_info['category'],
            'action': risk_info['action'],
            'top_drivers': top_drivers
        }
        
    sample_explanation = explain_applicant(0)
    print("\n--- Sample Applicant Explanation ---")
    print(json.dumps(sample_explanation, indent=4))
    
    return sample_explanation

if __name__ == '__main__':
    run_explainability()
