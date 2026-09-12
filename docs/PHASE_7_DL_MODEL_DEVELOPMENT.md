# Phase 7 — Deep Learning Model Development

## 1. Overview
In Phase 7, two distinct Deep Learning models were developed in **PyTorch**:
1. **Artificial Neural Network (ANN):** Trained on static normalized financial features.
2. **Sequential LSTM Network:** Trained on 12-month sequential repayment status trajectories (`m_1` to `m_12`).

- **DL Training Script:** [`src/train_dl.py`](../src/train_dl.py)
- **ANN Model Checkpoint:** [`models/ann_model.pt`](../models/ann_model.pt)
- **LSTM Model Checkpoint:** [`models/lstm_model.pt`](../models/lstm_model.pt)

---

## 2. Model Architectures

### A. Artificial Neural Network (ANN)
- **Input Dimension:** 29 normalized static financial features
- **Layer Breakdown:**
  - `Linear(29, 128)` $\to$ `BatchNorm1d(128)` $\to$ `ReLU()` $\to$ `Dropout(0.3)`
  - `Linear(128, 64)` $\to$ `BatchNorm1d(64)` $\to$ `ReLU()` $\to$ `Dropout(0.2)`
  - `Linear(64, 32)` $\to$ `ReLU()`
  - `Linear(32, 1)` $\to$ `Sigmoid()`
- **Optimizer:** AdamW ($lr = 0.003$) with weighted `BCEWithLogitsLoss`.

### B. Sequential LSTM Network
- **Input Dimension:** 12-month categorical status sequences ($S \in \mathbb{R}^{B \times 12}$)
- **Layer Breakdown:**
  - `Embedding(num_embeddings=4, embedding_dim=16)`
  - `LSTM(input_size=16, hidden_size=32, num_layers=2, batch_first=True, dropout=0.2)`
  - `Linear(32, 16)` $\to$ `ReLU()`
  - `Linear(16, 1)` $\to$ `Sigmoid()`
- **Optimizer:** AdamW ($lr = 0.005$).

---

## 3. Test Set Performance Comparison

| Model | Input Type | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **ANN** | Static Financials | 0.7253 | 0.6766 | 0.6975 | 0.6869 | 0.7943 |
| **LSTM** | 12m Repayment Trajectory | **0.9760** | **0.9842** | **0.9599** | **0.9719** | **0.9943** |

---

## 4. Key Behavioral Insight
- The **ANN model** provides high-value early-stage screening before loan issuance (when no repayment history exists yet).
- The **LSTM model** demonstrates exceptional predictive power ($0.9943$ ROC-AUC) for active loans by identifying early 30d/60d delinquency patterns.
