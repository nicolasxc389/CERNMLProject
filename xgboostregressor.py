import tkinter as tk
from tkinter import filedialog
import duckdb
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# ====================== FILE SELECTION ======================
parquet = tk.Tk()
parquet.withdraw()
file_path = filedialog.askopenfilename(
    title="Select a Parquet file", 
    filetypes=[("Parquet files", "*.parquet")]
)
if not file_path:
    print("No file selected.")
    exit()

con = duckdb.connect()

# Discover particles
query_discover = f"""
    SELECT column_name 
    FROM (DESCRIBE SELECT * FROM '{file_path}') 
    WHERE column_name ILIKE '%_ENDVERTEX_XERR' 
       OR column_name ILIKE '%_ENDVERTEX_YERR' 
       OR column_name ILIKE '%_ENDVERTEX_ZERR'
"""
columns_found = [col[0] for col in con.execute(query_discover).fetchall()]

prefixes = sorted({
    col.split('_ENDVERTEX_XERR')[0]
       .split('_ENDVERTEX_YERR')[0]
       .split('_ENDVERTEX_ZERR')[0] 
    for col in columns_found
})

print("\nFound particle prefixes:")
for p in prefixes:
    print(f"  - {p}")

particle = input('\nEnter exact particle prefix: ').strip()

# ====================== LOAD DATA ======================
query_main = f"""
    SELECT 
        {particle}_M,
        {particle}_PT, {particle}_PZ,
        {particle}_PX, {particle}_PY, {particle}_PE,
        {particle}_ENDVERTEX_XERR,
        {particle}_ENDVERTEX_YERR,
        {particle}_ENDVERTEX_ZERR
    FROM read_parquet('{file_path}')
    WHERE {particle}_M IS NOT NULL 
      AND {particle}_PX IS NOT NULL 
      AND {particle}_PE IS NOT NULL
"""
df = con.execute(query_main).df()
print(f"Loaded {len(df):,} events for {particle}")

# Compute invariant mass for peak estimation ONLY (not as feature)
df['m_inv'] = np.sqrt(np.maximum(0, 
    df[f'{particle}_PE']**2 - 
    (df[f'{particle}_PX']**2 + df[f'{particle}_PY']**2 + df[f'{particle}_PZ']**2)
))

# Estimate true peak from histogram of m_inv
hist, bin_edges = np.histogram(df['m_inv'], bins=300, 
                               range=(df['m_inv'].quantile(0.01), df['m_inv'].quantile(0.99)))
peak_bin = np.argmax(hist)
true_peak = (bin_edges[peak_bin] + bin_edges[peak_bin + 1]) / 2
print(f"Estimated true peak mass: {true_peak:.4f} MeV")

# ====================== FEATURES - NO LEAKAGE ======================
features = [
    f'{particle}_PT',
    f'{particle}_PZ',
    f'{particle}_PX',
    f'{particle}_PY',
    f'{particle}_ENDVERTEX_XERR',
    f'{particle}_ENDVERTEX_YERR',
    f'{particle}_ENDVERTEX_ZERR',
    # Removed PE, m_inv, and reconstructed M to prevent leakage
]

# Target = how much to correct the reconstructed mass
df['distortion'] = true_peak - df[f'{particle}_M']

X = df[features]
y = df['distortion']

print(f"Using {len(features)} features: {features}")

# ====================== TRAIN/TEST SPLIT ======================
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, df.index, test_size=0.25, random_state=42
)

# ====================== STRONGLY REGULARIZED XGBoost ======================
regressor = xgb.XGBRegressor(
    n_estimators=800,
    max_depth=4,              
    learning_rate=0.02,
    subsample=0.7,
    colsample_bytree=0.7,
    min_child_weight=10,
    gamma=0.5,
    reg_alpha=0.5,
    reg_lambda=2.0,
    random_state=42,
    n_jobs=-1,
    eval_metric='rmse',
    early_stopping_rounds=80
)

print("\nTraining XGBoost with strong regularization and early stopping...")

eval_set = [(X_train, y_train), (X_test, y_test)]
regressor.fit(
    X_train, y_train,
    eval_set=eval_set,
    verbose=False
)

# ====================== EVALUATION ======================
predicted_correction = regressor.predict(X_test)

original_mass_test = df.loc[idx_test, f'{particle}_M'].values
corrected_mass_test = original_mass_test + predicted_correction

sigma_before = np.std(original_mass_test)
sigma_after = np.std(corrected_mass_test)
improvement = ((sigma_before - sigma_after) / sigma_before) * 100 if sigma_before > 0 else 0

print("\n" + "="*70)
print("RESOLUTION SHARPENING RESULTS")
print("="*70)
print(f"Particle            : {particle}")
print(f"True peak           : {true_peak:.4f} MeV")
print(f"Original sigma      : {sigma_before:.4f} MeV")
print(f"Corrected sigma     : {sigma_after:.4f} MeV")
print(f"Improvement         : {improvement:.2f}%")
print(f"Test RMSE           : {np.sqrt(mean_squared_error(y_test, predicted_correction)):.4f} MeV")
print(f"Test R²             : {r2_score(y_test, predicted_correction):.4f}")
print("="*70)

# Feature importance
print("\nTop Feature Importances:")
importances = sorted(zip(features, regressor.feature_importances_), key=lambda x: x[1], reverse=True)
for feat, imp in importances[:8]:
    print(f"  {feat:35} {imp:.5f}")

# Plots
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(original_mass_test, bins=150, alpha=0.7, label='Original Reconstructed', density=True)
plt.hist(corrected_mass_test, bins=150, alpha=0.7, label='Corrected', density=True)
plt.axvline(true_peak, color='red', linestyle='--', linewidth=2, label='Estimated True Peak')
plt.xlabel('Mass (MeV)')
plt.ylabel('Density')
plt.title(f'{particle} Mass Resolution')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.hist(y_test - predicted_correction, bins=100, alpha=0.7, label='Residuals (True - Pred)')
plt.xlabel('Residual (MeV)')
plt.title('Residual Distribution')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{particle}_resolution_sharpening.png", dpi=300, bbox_inches='tight')
plt.show()

# Save summary
with open(f"{particle}_resolution_results.txt", "w") as f:
    f.write(f"Particle: {particle}\nTrue peak: {true_peak:.4f} MeV\n")
    f.write(f"Original sigma: {sigma_before:.4f}\nCorrected sigma: {sigma_after:.4f}\n")
    f.write(f"Improvement: {improvement:.2f}%\nR2: {r2_score(y_test, predicted_correction):.4f}\n")
