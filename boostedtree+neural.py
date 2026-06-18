import sys
import math
import tkinter as tk
from tkinter import filedialog
import duckdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from scipy.optimize import curve_fit

# ==========================================
# 1. FILE SELECTION & DATABASE CONNECTION
# ==========================================
print("Opening file dialog... Please select your Parquet file.")
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename(
    title="Select a Parquet file", 
    filetypes=[("Parquet files", "*.parquet")]
)

if not file_path:
    print("No file selected. Exiting.")
    sys.exit()

con = duckdb.connect()

# ==========================================
# 2. DATA EXTRACTION (THE WHOLE PICTURE)
# ==========================================
print("\nExtracting whole decay chain topology for B+ -> J/psi(mu+ mu-) K+ ...")

# This query extracts the complete decay tree topology while strictly omitting the B_M mass to prevent leakage.
# Adjust prefixes (e.g., 'B_', 'Jpsi_', 'K_', 'MuPlus_', 'MuMinus_') if your file uses a different naming scheme.
query = f"""
    SELECT 
        Bplus_PT AS b_pt, 
        SQRT(Bplus_PX*Bplus_PX + Bplus_PY*Bplus_PY + Bplus_PZ*Bplus_PZ) AS b_p_tot, 
        Bplus_FDCHI2_OWNPV AS b_fdchi2, 
        Bplus_IPCHI2_OWNPV AS b_ipchi2, 
        J_psi_1S_M AS jpsi_m, 
        J_psi_1S_PT AS jpsi_pt, 
        J_psi_1S_OWNPV_CHI2 AS jpsi_vchi2, 
        Kplus_PT AS k_pt, 
        Kplus_IPCHI2_OWNPV AS k_ipchi2, 
        Kplus_PIDK AS k_pidk, 
        Muplus_PT AS muplus_pt, 
        Muplus_IPCHI2_OWNPV AS muplus_ipchi2, 
        Muminus_PT AS muminus_pt, 
        Muminus_IPCHI2_OWNPV AS muminus_ipchi2, 
        SQRT(ABS(Bplus_PE*Bplus_PE - (Bplus_PX*Bplus_PX + Bplus_PY*Bplus_PY + Bplus_PZ*Bplus_PZ))) AS m_inv 
        FROM '{file_path}' 
    WHERE Bplus_PX IS NOT NULL 
        AND Bplus_PE IS NOT NULL 
        AND J_psi_1S_M IS NOT NULL 
        AND Kplus_PT IS NOT NULL 
        AND Muplus_PT IS NOT NULL 
        AND Muminus_PT IS NOT NULL;

"""
df_master = con.execute(query).df()

# ==========================================
# 3. ROBUST GAUSSIAN PEAK FINDER
# ==========================================
counts, bin_edges = np.histogram(df_master['m_inv'], bins=100)
bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

peak_idx = np.argmax(counts)
amp_guess = counts[peak_idx]
mean_guess = bin_centers[peak_idx]
sigma_guess = np.std(df_master['m_inv']) / 4.0

def gaussian(x, amp, mean, sigma):
    return amp * np.exp(-((x - mean) ** 2) / (2 * sigma ** 2))

print(f"Fitting mass peak to isolate Signal and Sidebands...")
try:
    popt, _ = curve_fit(gaussian, bin_centers, counts, p0=[amp_guess, mean_guess, sigma_guess])
    fit_mean, fit_sigma = popt[1], abs(popt[2])
    
    # 2-Sigma core for pure signal, 5+ Sigma out for background sidebands
    sig_low, sig_high = fit_mean - (2.0 * fit_sigma), fit_mean + (2.0 * fit_sigma)
    bkg_low, bkg_high = fit_mean - (5.0 * fit_sigma), fit_mean + (5.0 * fit_sigma)
    print(f"--> Success! Peak Mean: {fit_mean:.4f}, Sigma: {fit_sigma:.4f}")
except Exception as e:
    print(f"--> Fit failed ({e}). Falling back to robust percentiles.")
    sig_low, sig_high = np.percentile(df_master['m_inv'], [35, 65])
    bkg_low, bkg_high = np.percentile(df_master['m_inv'], [5, 95])

# Create Dataframes
signal_mask = (df_master['m_inv'] > sig_low) & (df_master['m_inv'] < sig_high)
bkg_mask = (df_master['m_inv'] < bkg_low) | (df_master['m_inv'] > bkg_high)

df_signal = df_master[signal_mask].copy()
df_bkg = df_master[bkg_mask].copy()

df_signal['label'] = 1
df_bkg['label'] = 0

df_train_ready = pd.concat([df_signal, df_bkg], ignore_index=True)

# ==========================================
# 4. MATRIX SEPARATION & DATA SPLITTING
# ==========================================
# Defining the complete feature set across the whole decay chain
features = [
    'b_pt', 'b_p_tot', 'b_fdchi2', 'b_ipchi2',
    'jpsi_m', 'jpsi_pt', 'jpsi_vchi2',
    'k_pt', 'k_ipchi2', 'k_pidk',
    'muplus_pt', 'muplus_ipchi2',
    'muminus_pt', 'muminus_ipchi2'
]

X = df_train_ready[features]
y = df_train_ready['label']

# Standardizing features is critical for Deep Learning convergence, though BDT ignores it
X_mean = X.mean()
X_std = X.std()
X_scaled = (X - X_mean) / X_std

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.30, random_state=42, stratify=y)

print(f"\n--- Data Split Diagnostics ---")
print(f"Train Size: {len(X_train)} | Test Size: {len(X_test)}")
print(f"Class Balance (Signal Ratio): {y_train.mean():.4f}")

# ==========================================
# 5. MODEL CHALLENGER 1: BOOSTED DECISION TREE
# ==========================================
print("\nTraining the Boosted Decision Tree (XGBoost)...")
bdt = xgb.XGBClassifier(
    n_estimators=150,
    max_depth=4,
    learning_rate=0.05,
    eval_metric="logloss",
    random_state=42
)
bdt.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=False)

bdt_train_preds = bdt.predict_proba(X_train)[:, 1]
bdt_test_preds = bdt.predict_proba(X_test)[:, 1]

# ==========================================
# 6. MODEL CHALLENGER 2: DEEP NEURAL NETWORK
# ==========================================
print("Training the Deep Neural Network (PyTorch)...")

# Convert pandas splits to PyTorch Tensors
X_train_t = torch.tensor(X_train.values, dtype=torch.float32)
y_train_t = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
X_test_t = torch.tensor(X_test.values, dtype=torch.float32)
y_test_t = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

class PhysicsClassifier(nn.Module):
    def __init__(self, input_dim):
        super(PhysicsClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.net(x)

dnn = PhysicsClassifier(input_dim=X_train.shape[1])
criterion = nn.BCELoss()
optimizer = optim.Adam(dnn.parameters(), lr=0.003)

dataset = TensorDataset(X_train_t, y_train_t)
loader = DataLoader(dataset, batch_size=256, shuffle=True)

dnn_train_losses = []
dnn_test_losses = []

dnn.train()
for epoch in range(40):
    epoch_loss = 0
    for batch_X, batch_y in loader:
        optimizer.zero_grad()
        predictions = dnn(batch_X)
        loss = criterion(predictions, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * batch_X.size(0)
    
    # Track evaluation loss per epoch
    dnn.eval()
    with torch.no_grad():
        test_loss = criterion(dnn(X_test_t), y_test_t).item()
    dnn_train_losses.append(epoch_loss / len(X_train))
    dnn_test_losses.append(test_loss)
    dnn.train()

dnn.eval()
with torch.no_grad():
    dnn_train_preds = dnn(X_train_t).numpy()
    dnn_test_preds = dnn(X_test_t).numpy()

# ==========================================
# 7. PERFORMANCE SHOWDOWN EVALUATION
# ==========================================
print(f"BDT Training AUC Score: {roc_auc_score(y_train, bdt_train_preds):.4f}")
print(f"BDT Testing AUC Score:  {roc_auc_score(y_test, bdt_test_preds):.4f}")
print("----------------------------------------------")
print(f"DNN Training AUC Score: {roc_auc_score(y_train, dnn_train_preds):.4f}")
print(f"DNN Testing AUC Score:  {roc_auc_score(y_test, dnn_test_preds):.4f}")
print("==============================================")

print("\n--- BDT Top Feature Importances ---")
bdt_importances = sorted(zip(features, bdt.feature_importances_), key=lambda x: x[1], reverse=True)
for col, imp in bdt_importances[:5]:
    print(f"  {col}: {imp:.4f}")

# ==========================================
# 8. VISUAL DIAGNOSTICS
# ==========================================
bdt_results = bdt.evals_result()

plt.figure(figsize=(12, 5))

# Plot 1: BDT Loss Curve
plt.subplot(1, 2, 1)
plt.plot(bdt_results['validation_0']['logloss'], label='BDT Train Loss')
plt.plot(bdt_results['validation_1']['logloss'], label='BDT Test Loss', linestyle='--')
plt.title('BDT Training Dynamics')
plt.xlabel('Iteration')
plt.ylabel('Logloss')
plt.legend()

# Plot 2: DNN Loss Curve
plt.subplot(1, 2, 2)
plt.plot(dnn_train_losses, label='DNN Train Loss', color='darkorange')
plt.plot(dnn_test_losses, label='DNN Test Loss', color='red', linestyle='--')
plt.title('DNN Training Dynamics')
plt.xlabel('Epoch')
plt.ylabel('Loss (Binary Cross Entropy)')
plt.legend()

plt.tight_layout()
plt.savefig('model_comparison_dynamics.png', dpi=200)
print("\nDiagnostics plot saved as 'model_comparison_dynamics.png'.")
plt.show()

# ==========================================
# 9. APPLYING DNN TO REAL DATA
# ==========================================
print("\nApplying DNN to filter the real data...")

# Scale the entire master dataset (including the peak region we masked earlier)
X_master_scaled = (df_master[features] - X_mean) / X_std
X_master_t = torch.tensor(X_master_scaled.values, dtype=torch.float32)

dnn.eval()
with torch.no_grad():
    df_master['dnn_score'] = dnn(X_master_t).numpy()

# Apply a strict cut (e.g., keeping only events with > 90% signal probability)
optimal_cut = 0.90 
df_clean_signal = df_master[df_master['dnn_score'] > optimal_cut]

# Plot the transformation
plt.figure(figsize=(8, 5))
plt.hist(df_master['m_inv'], bins=100, alpha=0.4, label='Raw Data (No Cut)', color='gray')
plt.hist(df_clean_signal['m_inv'], bins=100, alpha=0.8, label=f'Clean Data (DNN > {optimal_cut})', color='darkorange')
plt.title(f'Impact of Deep Learning Selection on $B^+ \\to J/\\psi K^+$ Mass Peak')
plt.xlabel('Invariant Mass [MeV]')
plt.ylabel('Candidates / Bin')
plt.axvline(x=np.average(df_master['m_inv']), color='red', linestyle='--', linewidth=1.5, label=f"Average Invariant Mass: {math.trunc(np.average(df_master['m_inv']))} MeV/c²")
plt.legend()
plt.savefig('real_data_dnn_selection.png', dpi=200)
plt.show()