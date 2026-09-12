import os
import json
import numpy as np
import pandas as pd
import joblib
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
)

from train_dl import FinancialANN, RepaymentLSTM

plt.style.use('ggplot')

def run_model_comparison(
    data_path='data/processed/processed_tensors.npz',
    repayment_path='data/raw/raw_repayment_history.csv',
    model_dir='models',
    output_dir='artifacts'
):
    """
    Phase 8: Model Comparison & Selection
    Evaluates Random Forest, XGBoost, ANN, and LSTM on Test set.
    Generates comparative ROC curves and selects champion model.
    """
    eval_plot_dir = os.path.join(output_dir, 'evaluation_plots')
    os.makedirs(eval_plot_dir, exist_ok=True)
    
    print(f"[EVALUATION] Loading test data from: {data_path}")
    data = np.load(data_path)
    X_test, y_test = data['X_test'], data['y_test']
    
    results = {}
    curves = {}
    
    # 1. Evaluate Random Forest
    rf_path = os.path.join(model_dir, 'random_forest.joblib')
    if os.path.exists(rf_path):
        rf_model = joblib.load(rf_path)
        rf_prob = rf_model.predict_proba(X_test)[:, 1]
        rf_pred = (rf_prob >= 0.5).astype(int)
        
        results['Random Forest'] = {
            'Accuracy': float(round(accuracy_score(y_test, rf_pred), 4)),
            'Precision': float(round(precision_score(y_test, rf_pred, zero_division=0), 4)),
            'Recall': float(round(recall_score(y_test, rf_pred, zero_division=0), 4)),
            'F1-Score': float(round(f1_score(y_test, rf_pred, zero_division=0), 4)),
            'ROC-AUC': float(round(roc_auc_score(y_test, rf_prob), 4))
        }
        curves['Random Forest'] = (y_test, rf_prob)
        
    # 2. Evaluate XGBoost
    xgb_path = os.path.join(model_dir, 'xgboost.joblib')
    if os.path.exists(xgb_path):
        xgb_model = joblib.load(xgb_path)
        xgb_prob = xgb_model.predict_proba(X_test)[:, 1]
        xgb_pred = (xgb_prob >= 0.5).astype(int)
        
        results['XGBoost'] = {
            'Accuracy': float(round(accuracy_score(y_test, xgb_pred), 4)),
            'Precision': float(round(precision_score(y_test, xgb_pred, zero_division=0), 4)),
            'Recall': float(round(recall_score(y_test, xgb_pred, zero_division=0), 4)),
            'F1-Score': float(round(f1_score(y_test, xgb_pred, zero_division=0), 4)),
            'ROC-AUC': float(round(roc_auc_score(y_test, xgb_prob), 4))
        }
        curves['XGBoost'] = (y_test, xgb_prob)
        
    # 3. Evaluate ANN
    ann_path = os.path.join(model_dir, 'ann_model.pt')
    if os.path.exists(ann_path):
        ann_model = FinancialANN(input_dim=X_test.shape[1])
        ann_model.load_state_dict(torch.load(ann_path))
        ann_model.eval()
        
        with torch.no_grad():
            ann_logits = ann_model(torch.tensor(X_test, dtype=torch.float32)).squeeze(-1)
            ann_prob = torch.sigmoid(ann_logits).numpy()
            ann_pred = (ann_prob >= 0.5).astype(int)
            
        results['ANN'] = {
            'Accuracy': float(round(accuracy_score(y_test, ann_pred), 4)),
            'Precision': float(round(precision_score(y_test, ann_pred, zero_division=0), 4)),
            'Recall': float(round(recall_score(y_test, ann_pred, zero_division=0), 4)),
            'F1-Score': float(round(f1_score(y_test, ann_pred, zero_division=0), 4)),
            'ROC-AUC': float(round(roc_auc_score(y_test, ann_prob), 4))
        }
        curves['ANN'] = (y_test, ann_prob)
        
    # 4. Evaluate LSTM
    lstm_path = os.path.join(model_dir, 'lstm_model.pt')
    if os.path.exists(lstm_path) and os.path.exists(repayment_path):
        df_rep = pd.read_csv(repayment_path)
        seq_cols = [f"m_{m}" for m in range(1, 13)]
        seq_matrix = df_rep[seq_cols].values
        targets = df_rep['loan_status'].values
        
        from sklearn.model_selection import train_test_split
        idx_train_val, idx_test = train_test_split(np.arange(len(df_rep)), test_size=0.15, random_state=42, stratify=targets)
        
        lstm_seq_test = torch.tensor(seq_matrix[idx_test], dtype=torch.long)
        y_lstm_test = targets[idx_test]
        
        lstm_model = RepaymentLSTM(num_statuses=4, embedding_dim=16, hidden_dim=32, num_layers=2)
        lstm_model.load_state_dict(torch.load(lstm_path))
        lstm_model.eval()
        
        with torch.no_grad():
            lstm_logits = lstm_model(lstm_seq_test).squeeze(-1)
            lstm_prob = torch.sigmoid(lstm_logits).numpy()
            lstm_pred = (lstm_prob >= 0.5).astype(int)
            
        results['LSTM'] = {
            'Accuracy': float(round(accuracy_score(y_lstm_test, lstm_pred), 4)),
            'Precision': float(round(precision_score(y_lstm_test, lstm_pred, zero_division=0), 4)),
            'Recall': float(round(recall_score(y_lstm_test, lstm_pred, zero_division=0), 4)),
            'F1-Score': float(round(f1_score(y_lstm_test, lstm_pred, zero_division=0), 4)),
            'ROC-AUC': float(round(roc_auc_score(y_lstm_test, lstm_prob), 4))
        }
        curves['LSTM'] = (y_lstm_test, lstm_prob)
        
    # Save Comparison JSON
    json_path = os.path.join(output_dir, 'model_comparison.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"\n[MODEL COMPARISON SUMMARY]")
    df_res = pd.DataFrame(results).T
    print(df_res.to_string())
    
    # 5. Plot Comparative ROC Curves
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {'Random Forest': '#2980b9', 'XGBoost': '#27ae60', 'ANN': '#8e44ad', 'LSTM': '#e74c3c'}
    
    for model_name, (y_t, y_p) in curves.items():
        fpr, tpr, _ = roc_curve(y_t, y_p)
        auc_val = results[model_name]['ROC-AUC']
        ax.plot(fpr, tpr, label=f"{model_name} (AUC = {auc_val:.4f})", color=colors.get(model_name, 'black'), lw=2)
        
    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Chance')
    ax.set_title('Comparative ROC Curves (Test Set)', fontsize=14, fontweight='bold')
    ax.set_xlabel('False Positive Rate (1 - Specificity)')
    ax.set_ylabel('True Positive Rate (Recall)')
    ax.legend(loc='lower right')
    plt.tight_layout()
    
    roc_plot_path = os.path.join(eval_plot_dir, 'roc_comparison.png')
    plt.savefig(roc_plot_path, dpi=300)
    plt.close()
    print(f"[SUCCESS] Saved ROC Curves plot to: {roc_plot_path}")
    print(f"[SUCCESS] Saved model metrics comparison JSON to: {json_path}")

if __name__ == '__main__':
    run_model_comparison()
