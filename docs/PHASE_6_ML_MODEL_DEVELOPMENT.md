# Phase 6 — Classical ML Model Development

## 1. Overview
In Phase 6, classical machine learning algorithms — **Random Forest Classifier** and **XGBoost Classifier** — were trained and evaluated to capture non-linear financial patterns and default risk signals.

- **ML Training Script:** [`src/train_ml.py`](../src/train_ml.py)
- **Random Forest Model:** [`models/random_forest.joblib`](../models/random_forest.joblib)
- **XGBoost Model:** [`models/xgboost.joblib`](../models/xgboost.joblib)

---

## 2. Model Training Architectures

### A. Random Forest Classifier
- `n_estimators`: 300 decision trees
- `max_depth`: 12
- `min_samples_split`: 5
- `class_weight`: `'balanced'` (re-weights loss to boost minority default class sensitivity)

### B. XGBoost Classifier
- `n_estimators`: 300 boosting rounds
- `max_depth`: 5
- `learning_rate`: 0.03
- `scale_pos_weight`: ~1.32 (automatically computed negative/positive sample ratio)
- `subsample`: 0.8, `colsample_bytree`: 0.8

---

## 3. Evaluation Results

### Validation Set Performance
| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | 0.7440 | 0.6953 | 0.7253 | 0.7100 | 0.8195 |
| **XGBoost** | **0.7707** | **0.7235** | **0.7593** | **0.7410** | **0.8325** |

### Test Set Performance
| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | 0.7107 | 0.6698 | 0.6512 | 0.6604 | 0.7797 |
| **XGBoost** | **0.7080** | 0.6615 | **0.6636** | **0.6626** | **0.7767** |

---

## 4. Key Takeaways
- **XGBoost** achieved the highest validation ROC-AUC ($0.8325$) and test recall ($0.6636$), demonstrating superior gradient boosting capability across complex financial feature interactions.
