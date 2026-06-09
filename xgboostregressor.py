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
    print("No file selected. Exiting.")
    exit()

con = duckdb.connect()

# ====================== DISCOVER PARTICLES ======================
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

particle = input('\nEnter the exact particle prefix: ').strip()

# ====================== LOAD DATA ======================
# Load kinematics + vertex errors
query_main = f"""
    SELECT 
        {particle}_M,
        {particle}_PT,
        {particle}_PZ,
        {particle}_PX,
        {particle}_PY,
        {particle}_PE,
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

# ====================== COMPUTE INVARIANT MASS (m_inv) ======================
# Use the 4-momentum to compute true invariant mass
df['m_inv'] = np.sqrt(
    np.maximum(0, 
        df[f'{particle}_PE']**2 - 
        (df[f'{particle}_PX']**2 + df[f'{particle}_PY']**2 + df[f'{particle}_PZ']**2)
    )
)

# Convert the pandas dataframe to a NumPy array for histogramming to make the graph easier to analyze
mass = df['m_inv'].to_numpy()

# Estimate true peak (PDG-like mass) from the distribution
hist, bin_edges = np.histogram(df['m_inv'], bins=200, range=(df['m_inv'].quantile(0.01), df['m_inv'].quantile(0.99)))
peak_bin = np.argmax(hist)
true_peak = (bin_edges[peak_bin] + bin_edges[peak_bin + 1]) / 2

print(f"Estimated true peak mass: {true_peak:.4f} MeV")

# ====================== FEATURES & TARGET ======================
features = [
    f'{particle}_M',           # reconstructed mass
    f'{particle}_PT',
    f'{particle}_PZ',
    f'{particle}_ENDVERTEX_XERR',
    f'{particle}_ENDVERTEX_YERR',
    f'{particle}_ENDVERTEX_ZERR',
    'm_inv'                    # invariant mass as extra feature
]

# Target = distortion to correct (how much to add to reconstructed mass)
df['distortion'] = true_peak - df[f'{particle}_M']

X = df[features]
y = df['distortion']

# ====================== TRAIN / TEST SPLIT ======================
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, df.index, test_size=0.2, random_state=42
)

# ====================== XGBoost ======================
regressor = xgb.XGBRegressor(
    n_estimators=400,
    max_depth=7,
    learning_rate=0.05,
    subsample=0.85,
    colsample_bytree=0.85,
    random_state=42,
    n_jobs=-1,
    eval_metric='rmse'
)

print("\nTraining XGBoost for mass resolution correction...")
regressor.fit(X_train, y_train)

# ====================== EVALUATION ======================
predicted_correction = regressor.predict(X_test)

original_mass_test = df.loc[idx_test, f'{particle}_M'].values
corrected_mass_test = original_mass_test + predicted_correction

sigma_before = np.std(original_mass_test)
sigma_after = np.std(corrected_mass_test)
improvement = ((sigma_before - sigma_after) / sigma_before) * 100 if sigma_before > 0 else 0

print("\n" + "="*60)
print("RESOLUTION SHARPENING RESULTS")
print("="*60)
print(f"Particle: {particle}")
print(f"True peak ≈ {true_peak:.4f} MeV")
print(f"Original sigma:  {sigma_before:.4f} MeV")
print(f"Corrected sigma: {sigma_after:.4f} MeV")
print(f"Improvement:     {improvement:.2f}%")
print(f"Regression RMSE: {np.sqrt(mean_squared_error(y_test, predicted_correction)):.4f} MeV")
print(f"R² score:        {r2_score(y_test, predicted_correction):.4f}")
print("="*60)

# Save results
with open(f"{particle}_resolution_results.txt", "w") as f:
    f.write(f"Particle: {particle}\n")
    f.write(f"True peak: {true_peak:.4f} MeV\n")
    f.write(f"Original sigma: {sigma_before:.4f} MeV\n")
    f.write(f"Corrected sigma: {sigma_after:.4f} MeV\n")
    f.write(f"Improvement: {improvement:.2f}%\n")
    f.write(f"RMSE: {np.sqrt(mean_squared_error(y_test, predicted_correction)):.4f} MeV\n")

# Optional: Plot before/after
plt.figure(figsize=(10, 6))
plt.hist(original_mass_test, bins=100, alpha=0.7, label='Original', density=True)
plt.hist(corrected_mass_test, bins=100, alpha=0.7, label='Corrected', density=True)
plt.xlim(float(np.min(mass)), float(np.max(mass)))
plt.axvline(true_peak, color='red', linestyle='--', label=f"True peak: {true_peak:.4f} MeV\n")
plt.xlabel('Mass (MeV)')
plt.ylabel('Density')
plt.title(f'{particle} Mass Resolution Sharpening')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(f"{particle}_mass_distribution.png", dpi=300, bbox_inches='tight')
plt.show()
