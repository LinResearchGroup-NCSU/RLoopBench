import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    accuracy_score, confusion_matrix,
    roc_auc_score, average_precision_score
)
import pandas as pd
import copy
import joblib
import os

# ============================
# 1. Setup and Directory Creation
# ============================
output_dir = "linear_probe"
os.makedirs(output_dir, exist_ok=True)
print(f">>> Output directory initialized: {output_dir}/")

# Set seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f">>> Using device: {device}")

# ============================
# 2. Load and Prepare Data
# ============================
print("\nLoading datasets...")
try:
    # Load raw .npy files
    pos_train = np.load("pos_training_3mer.npy")
    neg_train = np.load("neg_training_3mer.npy")
    pos_eval = np.load("pos_evaluation_3mer.npy")
    neg_eval = np.load("neg_evaluation_3mer.npy")
    pos_test = np.load("pos_testing_3mer.npy")
    neg_test = np.load("neg_testing_3mer.npy")
except FileNotFoundError as e:
    print(f"Error: Data files not found! {e}")
    exit()

# Helper function to stack positive and negative samples
def build_xy(pos, neg):
    X = np.vstack([pos, neg])
    y = np.array([1]*len(pos) + [0]*len(neg))
    return X, y

X_train_raw, y_train = build_xy(pos_train, neg_train)
X_eval_raw, y_eval = build_xy(pos_eval, neg_eval)
X_test_raw, y_test = build_xy(pos_test, neg_test)

# Standardization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_eval = scaler.transform(X_eval_raw)
X_test = scaler.transform(X_test_raw)

# Save the scaler for future inference
joblib.dump(scaler, f"{output_dir}/scaler.joblib")

# ============================
# 3. PyTorch Data Pipeline
# ============================
def create_loader(X, y, shuffle=False, batch_size=64):
    dataset = TensorDataset(
        torch.FloatTensor(X),
        torch.FloatTensor(y).unsqueeze(1)
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

train_loader = create_loader(X_train, y_train, shuffle=True)
eval_loader = create_loader(X_eval, y_eval)
test_loader = create_loader(X_test, y_test)

print(f"Train shape: {X_train.shape}, Input Dim: {X_train.shape[1]}")

# ============================
# 4. Model Architecture
# ============================
class LinearProbe(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)

# ============================
# 5. Training and Evaluation Functions
# ============================
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader):
    model.eval()
    all_probs, all_labels = [], []
    
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs.flatten())
            all_labels.extend(y.numpy().flatten())

    all_probs = np.array(all_probs)
    all_preds = (all_probs > 0.5).astype(int)
    all_labels = np.array(all_labels)

    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
    
    metrics = {
        "Accuracy": accuracy_score(all_labels, all_preds),
        "Precision": precision_score(all_labels, all_preds, zero_division=0),
        "Recall": recall_score(all_labels, all_preds),
        "F1": f1_score(all_labels, all_preds),
        "Specificity": tn / (tn + fp + 1e-7),
        "AUC": roc_auc_score(all_labels, all_probs),
        "PRAUC": average_precision_score(all_labels, all_probs)
    }
    return metrics

# ============================
# 6. Hyperparameter Grid Search
# ============================
learning_rates = [1e-4, 1e-3, 1e-2]
weight_decays = [0, 1e-4, 1e-2]
max_epochs = 100
patience = 10

results = []
best_global_f1 = -1
best_model_state = None
best_config_summary = None

print("\n>>> Starting Hyperparameter Search...")

for lr in learning_rates:
    for wd in weight_decays:
        print(f"Testing: lr={lr}, wd={wd}...", end=" ")
        
        input_dim = X_train.shape[1]
        model = LinearProbe(input_dim).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        criterion = nn.BCEWithLogitsLoss()

        best_val_f1_curr = -1
        patience_counter = 0
        best_state_curr = None
        last_epoch = 0

        for epoch in range(max_epochs):
            train_one_epoch(model, train_loader, optimizer, criterion)
            metrics = evaluate(model, eval_loader)
            
            val_f1 = metrics["F1"]
            if val_f1 > best_val_f1_curr:
                best_val_f1_curr = val_f1
                best_state_curr = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
            
            last_epoch = epoch + 1
            if patience_counter >= patience:
                break

        # CRITICAL: Load the best state for this config before logging results
        model.load_state_dict(best_state_curr)
        final_eval_metrics = evaluate(model, eval_loader)
        
        print(f"Done (Epochs: {last_epoch}, Best Val F1: {best_val_f1_curr:.4f})")

        res_entry = {
            "lr": lr,
            "weight_decay": wd,
            "epochs_trained": last_epoch,
            **final_eval_metrics
        }
        results.append(res_entry)

        # Update global best model
        if final_eval_metrics["F1"] > best_global_f1:
            best_global_f1 = final_eval_metrics["F1"]
            best_model_state = best_state_curr
            best_config_summary = res_entry

# ============================
# 7. Final Evaluation and Saving
# ============================
# Save hyperparameter logs
df_results = pd.DataFrame(results)
df_results.to_csv(f"{output_dir}/hyperparameter_search_results.csv", index=False)

print("\n" + "="*40)
print("BEST CONFIGURATION FOUND:")
print(pd.Series(best_config_summary))
print("="*40)

# Final Test set evaluation
final_model = LinearProbe(X_train.shape[1]).to(device)
final_model.load_state_dict(best_model_state)
test_metrics = evaluate(final_model, test_loader)

print("\n>>> Final Test Set Results:")
for k, v in test_metrics.items():
    print(f"{k:15}: {v:.4f}")

# Model Persistence
torch.save(best_model_state, f"{output_dir}/best_linear_probe.pt")
pd.DataFrame([test_metrics]).to_csv(f"{output_dir}/final_test_metrics.csv", index=False)

# Log summary info
with open(f"{output_dir}/model_summary.txt", "w") as f:
    f.write(f"Input Dimension: {X_train.shape[1]}\n")
    f.write(f"Total Parameters: {X_train.shape[1] + 1}\n")
    f.write(f"Best Hyperparameters: {str(best_config_summary)}\n")
    f.write(f"Final Test Metrics: {str(test_metrics)}\n")

print(f"\nAll outputs saved successfully to: {output_dir}/")
