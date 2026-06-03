import sys
import tkinter as tk
from tkinter import filedialog
import duckdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
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
# 2. USER PARTICLE SELECTION
# ==========================================
# Dynamic SQL to discover available columns
query_discover = f"SELECT column_name FROM (DESCRIBE SELECT * FROM '{file_path}') WHERE column_name ILIKE '%_PX%'"
columns_found = [col[0] for col in con.execute(query_discover).fetchall()]

print("\nFound the following available particle prefixes:")
prefixes = sorted(list(set([col.split('_PX')[0] for col in columns_found])))
for p in prefixes:
    print(f"  - {p}")

particle = input('\nEnter the exact particle prefix you want to analyze: ').strip()

# ==========================================
# 3. DATA EXTRACTION & KINEMATICS
# ==========================================
print(f"\nExtracting and calculating physics variables for {particle}...")
query = f"""
    SELECT 
        SQRT({particle}_PX^2 + {particle}_PY^2 + {particle}_PZ^2) AS p_tot,
        {particle}_PT as pt,
        SQRT(GREATEST(0, {particle}_PE^2 - ({particle}_PX^2 + {particle}_PY^2 + {particle}_PZ^2))) AS m_inv
    FROM '{file_path}'
    WHERE {particle}_PX IS NOT NULL AND {particle}_PY IS NOT NULL 
      AND {particle}_PZ IS NOT NULL AND {particle}_PE IS NOT NULL
"""
df_master = con.execute(query).df()

# ==========================================
# 4. ROBUST GAUSSIAN PEAK FINDER
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
# 5. MATRIX SEPARATION & DATA SPLITTING
# ==========================================
features = ['p_tot', 'pt']  # Mass 'm_inv' is strictly omitted to prevent leakage
X = df_train_ready[features]
y = df_train_ready['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)

print(f"\n--- Data Split Diagnostics ---")
print(f"Train Size: {len(X_train)} | Test Size: {len(X_test)}")
print(f"Class Balance (Signal Ratio) in Train: {y_train.mean():.4f}")

# ==========================================
# 6. BDT TRAINING
# ==========================================
print("\nTraining the Boosted Decision Tree...")
bdt = xgb.XGBClassifier(
    n_estimators=150,
    max_depth=3,
    learning_rate=0.05,
    eval_metric="logloss",
    random_state=42
)

bdt.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=False)

# ==========================================
# 7. PERFORMANCE & FEATURE IMPORTANCE
# ==========================================
train_preds = bdt.predict_proba(X_train)[:, 1]
test_preds = bdt.predict_proba(X_test)[:, 1]

print("\n--- Model Performance Evaluation ---")
print(f"Training AUC Score: {roc_auc_score(y_train, train_preds):.4f}")
print(f"Testing AUC Score:  {roc_auc_score(y_test, test_preds):.4f}")

print("\n--- Feature Importances ---")
for col, imp in zip(features, bdt.feature_importances_):
    print(f"  {col}: {imp:.4f}")

# ==========================================
# 8. VISUAL DIAGNOSTICS
# ==========================================
# Figure 1: Loss Curve
results = bdt.evals_result()
plt.figure(figsize=(6, 4))
plt.plot(results['validation_0']['logloss'], label='Train Loss')
plt.plot(results['validation_1']['logloss'], label='Test Loss', linestyle='--')
plt.title(f'BDT Loss Curve ({particle})')
plt.xlabel('Iteration')
plt.ylabel('Logloss')
plt.legend()
plt.savefig(f'{particle}_loss_curve.png', dpi=200)

# Figure 2: Kinematic Distribution Overlap (The Explainer Plot)
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.hist(df_train_ready[df_train_ready['label']==1]['p_tot'], bins=50, alpha=0.5, label='Signal', density=True)
plt.hist(df_train_ready[df_train_ready['label']==0]['p_tot'], bins=50, alpha=0.5, label='Background', density=True)
plt.title(f'{particle} $P_{{tot}}$ Distribution')
plt.legend()

plt.subplot(1, 2, 2)
plt.hist(df_train_ready[df_train_ready['label']==1]['pt'], bins=50, alpha=0.5, label='Signal', density=True)
plt.hist(df_train_ready[df_train_ready['label']==0]['pt'], bins=50, alpha=0.5, label='Background', density=True)
plt.title(f'{particle} $P_T$ Distribution')
plt.legend()

plt.tight_layout()
plt.savefig(f'{particle}_kinematics.png', dpi=200)
print(f"\nVisualizations saved: '{particle}_loss_curve.png' and '{particle}_kinematics.png'.")
plt.show()
