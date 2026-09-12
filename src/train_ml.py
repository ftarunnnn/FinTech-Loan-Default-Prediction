import os
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
)

def evaluate_predictions(y_true, y_pred, y_prob, name="Model"):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_prob)
    cm = confusion_matrix(y_true, y_pred).tolist()
    
    print(f"\n--- {name} Metrics ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC-AUC:   {auc:.4f}")
    print(f"Confusion Matrix: TN={cm[0][0]}, FP={cm[0][1]}, FN={cm[1][0]}, TP={cm[1][1]}")
    
    return {
        'Accuracy': float(round(acc, 4)),
        'Precision': float(round(prec, 4)),
        'Recall': float(round(rec, 4)),
        'F1-Score': float(round(f1, 4)),
        'ROC-AUC': float(round(auc, 4)),
        'Confusion_Matrix': cm
    }

def train_ml_models(data_path='data/processed/processed_tensors.npz', model_dir='models'):
    """
    Phase 6: ML Model Development
    Trains Random Forest and XGBoost classifiers.
    """
    os.makedirs(model_dir, exist_ok=True)
    print(f"[ML TRAINING] Loading processed tensors from: {data_path}")
    
    data = np.load(data_path)
    X_train, y_train = data['X_train'], data['y_train']
    X_val, y_val = data['X_val'], data['y_val']
    X_test, y_test = data['X_test'], data['y_test']
    
    # 1. Random Forest Classifier
    print("\n[TRAINING] Random Forest Classifier...")
    rf_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_split=5,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    
    rf_val_prob = rf_model.predict_proba(X_val)[:, 1]
    rf_val_pred = (rf_val_prob >= 0.5).astype(int)
    rf_val_metrics = evaluate_predictions(y_val, rf_val_pred, rf_val_prob, name="Random Forest (Val)")
    
    rf_test_prob = rf_model.predict_proba(X_test)[:, 1]
    rf_test_pred = (rf_test_prob >= 0.5).astype(int)
    rf_test_metrics = evaluate_predictions(y_test, rf_test_pred, rf_test_prob, name="Random Forest (Test)")
    
    rf_path = os.path.join(model_dir, 'random_forest.joblib')
    joblib.dump(rf_model, rf_path)
    print(f"[SUCCESS] Saved Random Forest model to: {rf_path}")
    
    # 2. XGBoost Classifier
    print("\n[TRAINING] XGBoost Classifier...")
    pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
    xgb_model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.03,
        scale_pos_weight=pos_weight,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss',
        n_jobs=-1
    )
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    
    xgb_val_prob = xgb_model.predict_proba(X_val)[:, 1]
    xgb_val_pred = (xgb_val_prob >= 0.5).astype(int)
    xgb_val_metrics = evaluate_predictions(y_val, xgb_val_pred, xgb_val_prob, name="XGBoost (Val)")
    
    xgb_test_prob = xgb_model.predict_proba(X_test)[:, 1]
    xgb_test_pred = (xgb_test_prob >= 0.5).astype(int)
    xgb_test_metrics = evaluate_predictions(y_test, xgb_test_pred, xgb_test_prob, name="XGBoost (Test)")
    
    xgb_path = os.path.join(model_dir, 'xgboost.joblib')
    joblib.dump(xgb_model, xgb_path)
    print(f"[SUCCESS] Saved XGBoost model to: {xgb_path}")
    
    return {
        'RandomForest': rf_test_metrics,
        'XGBoost': xgb_test_metrics
    }

if __name__ == '__main__':
    train_ml_models()
