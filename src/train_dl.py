import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# Set seed
torch.manual_seed(42)
np.random.seed(42)

# ----------------------------------------------------
# 1. PyTorch ANN Architecture
# ----------------------------------------------------
class FinancialANN(nn.Module):
    def __init__(self, input_dim):
        super(FinancialANN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        return self.net(x)

# ----------------------------------------------------
# 2. PyTorch LSTM Architecture
# ----------------------------------------------------
class RepaymentLSTM(nn.Module):
    def __init__(self, num_statuses=4, embedding_dim=16, hidden_dim=32, num_layers=2):
        super(RepaymentLSTM, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=num_statuses, embedding_dim=embedding_dim)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0
        )
        self.fc1 = nn.Linear(hidden_dim, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 1)
        
    def forward(self, x):
        # x shape: [batch_size, seq_len=12]
        embedded = self.embedding(x) # [batch_size, 12, embedding_dim]
        lstm_out, (hn, cn) = self.lstm(embedded)
        # Use final hidden state
        last_out = lstm_out[:, -1, :]
        out = self.relu(self.fc1(last_out))
        logits = self.fc2(out)
        return logits

def evaluate_pytorch_model(model, dataloader, is_lstm=False):
    model.eval()
    y_true_list = []
    y_prob_list = []
    
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            logits = model(batch_x)
            probs = torch.sigmoid(logits).squeeze(-1)
            y_true_list.extend(batch_y.numpy())
            y_prob_list.extend(probs.numpy())
            
    y_true = np.array(y_true_list)
    y_prob = np.array(y_prob_list)
    y_pred = (y_prob >= 0.5).astype(int)
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_prob)
    cm = confusion_matrix(y_true, y_pred).tolist()
    
    return {
        'Accuracy': float(round(acc, 4)),
        'Precision': float(round(prec, 4)),
        'Recall': float(round(rec, 4)),
        'F1-Score': float(round(f1, 4)),
        'ROC-AUC': float(round(auc, 4)),
        'Confusion_Matrix': cm,
        'y_prob': y_prob
    }

def train_dl_models(data_path='data/processed/processed_tensors.npz', repayment_path='data/raw/raw_repayment_history.csv', model_dir='models'):
    """
    Phase 7: DL Model Development
    Trains Artificial Neural Network (ANN) and Sequential LSTM models in PyTorch.
    """
    os.makedirs(model_dir, exist_ok=True)
    print(f"[DL TRAINING] Loading processed tensors from: {data_path}")
    
    data = np.load(data_path)
    X_train, y_train = data['X_train'], data['y_train']
    X_val, y_val = data['X_val'], data['y_val']
    X_test, y_test = data['X_test'], data['y_test']
    
    # Convert static features to PyTorch Tensors
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    # ----------------------------------------------------
    # Train ANN Model
    # ----------------------------------------------------
    print("\n[TRAINING] PyTorch Artificial Neural Network (ANN)...")
    ann_model = FinancialANN(input_dim=X_train.shape[1])
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.32]))
    optimizer = optim.AdamW(ann_model.parameters(), lr=0.003, weight_decay=1e-4)
    
    epochs = 35
    best_val_auc = 0.0
    
    for epoch in range(epochs):
        ann_model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            logits = ann_model(batch_x).squeeze(-1)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            
        val_metrics = evaluate_pytorch_model(ann_model, val_loader)
        if val_metrics['ROC-AUC'] > best_val_auc:
            best_val_auc = val_metrics['ROC-AUC']
            torch.save(ann_model.state_dict(), os.path.join(model_dir, 'ann_model.pt'))
            
    # Load best ANN checkpoint for evaluation
    ann_model.load_state_dict(torch.load(os.path.join(model_dir, 'ann_model.pt')))
    ann_test_metrics = evaluate_pytorch_model(ann_model, test_loader)
    
    print(f"--- ANN (Test) Metrics ---")
    print(f"Accuracy:  {ann_test_metrics['Accuracy']:.4f}")
    print(f"Precision: {ann_test_metrics['Precision']:.4f}")
    print(f"Recall:    {ann_test_metrics['Recall']:.4f}")
    print(f"F1-Score:  {ann_test_metrics['F1-Score']:.4f}")
    print(f"ROC-AUC:   {ann_test_metrics['ROC-AUC']:.4f}")
    print(f"[SUCCESS] Saved PyTorch ANN model to: {os.path.join(model_dir, 'ann_model.pt')}")
    
    # ----------------------------------------------------
    # Train Sequential LSTM Model
    # ----------------------------------------------------
    print("\n[TRAINING] PyTorch Repayment Trajectory LSTM...")
    df_rep = pd.read_csv(repayment_path)
    seq_cols = [f"m_{m}" for m in range(1, 13)]
    seq_matrix = df_rep[seq_cols].values
    targets = df_rep['loan_status'].values
    
    # Use matching train/val/test splits
    # Same indices as feature_engineering
    n_samples = len(df_rep)
    np.random.seed(42)
    indices = np.arange(n_samples)
    
    from sklearn.model_selection import train_test_split
    idx_train_val, idx_test = train_test_split(indices, test_size=0.15, random_state=42, stratify=targets)
    idx_train, idx_val = train_test_split(idx_train_val, test_size=0.17647, random_state=42, stratify=targets[idx_train_val])
    
    lstm_train_ds = TensorDataset(torch.tensor(seq_matrix[idx_train], dtype=torch.long), torch.tensor(targets[idx_train], dtype=torch.float32))
    lstm_val_ds = TensorDataset(torch.tensor(seq_matrix[idx_val], dtype=torch.long), torch.tensor(targets[idx_val], dtype=torch.float32))
    lstm_test_ds = TensorDataset(torch.tensor(seq_matrix[idx_test], dtype=torch.long), torch.tensor(targets[idx_test], dtype=torch.float32))
    
    lstm_train_loader = DataLoader(lstm_train_ds, batch_size=64, shuffle=True)
    lstm_val_loader = DataLoader(lstm_val_ds, batch_size=128, shuffle=False)
    lstm_test_loader = DataLoader(lstm_test_ds, batch_size=128, shuffle=False)
    
    lstm_model = RepaymentLSTM(num_statuses=4, embedding_dim=16, hidden_dim=32, num_layers=2)
    lstm_optimizer = optim.AdamW(lstm_model.parameters(), lr=0.005, weight_decay=1e-4)
    lstm_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.32]))
    
    best_lstm_auc = 0.0
    for epoch in range(25):
        lstm_model.train()
        for batch_x, batch_y in lstm_train_loader:
            lstm_optimizer.zero_grad()
            logits = lstm_model(batch_x).squeeze(-1)
            loss = lstm_criterion(logits, batch_y)
            loss.backward()
            lstm_optimizer.step()
            
        val_m = evaluate_pytorch_model(lstm_model, lstm_val_loader, is_lstm=True)
        if val_m['ROC-AUC'] > best_lstm_auc:
            best_lstm_auc = val_m['ROC-AUC']
            torch.save(lstm_model.state_dict(), os.path.join(model_dir, 'lstm_model.pt'))
            
    lstm_model.load_state_dict(torch.load(os.path.join(model_dir, 'lstm_model.pt')))
    lstm_test_metrics = evaluate_pytorch_model(lstm_model, lstm_test_loader, is_lstm=True)
    
    print(f"--- LSTM (Test) Metrics ---")
    print(f"Accuracy:  {lstm_test_metrics['Accuracy']:.4f}")
    print(f"Precision: {lstm_test_metrics['Precision']:.4f}")
    print(f"Recall:    {lstm_test_metrics['Recall']:.4f}")
    print(f"F1-Score:  {lstm_test_metrics['F1-Score']:.4f}")
    print(f"ROC-AUC:   {lstm_test_metrics['ROC-AUC']:.4f}")
    print(f"[SUCCESS] Saved PyTorch LSTM model to: {os.path.join(model_dir, 'lstm_model.pt')}")
    
    # Remove y_prob before returning
    ann_test_metrics.pop('y_prob', None)
    lstm_test_metrics.pop('y_prob', None)
    
    return {
        'ANN': ann_test_metrics,
        'LSTM': lstm_test_metrics
    }

if __name__ == '__main__':
    train_dl_models()
