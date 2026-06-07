import tkinter as tk
from tkinter import filedialog
import duckdb
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

#Select the parquet file
parquet = tk.Tk()
parquet.withdraw()
file_path = filedialog.askopenfilename(title="Select a Parquet file", filetypes =[("Parquet files", "*.parquet")])

# Connect to the file
con = duckdb.connect()

query_discover = f"""
    SELECT column_name 
    FROM (DESCRIBE SELECT * FROM '{file_path}') 
    WHERE column_name ILIKE '%_ENDVERTEX_XERR' 
       OR column_name ILIKE '%_ENDVERTEX_YERR' 
       OR column_name ILIKE '%_ENDVERTEX_ZERR'
"""

# Fetch columns from the database (assuming 'con' and 'query_discover' are defined)
columns_found = [col[0] for col in con.execute(query_discover).fetchall()]

print("\nFound the following available particle prefixes:")

# Using a set comprehension simplifies the syntax and removes duplicates automatically
prefixes = sorted({
    col.split('_ENDVERTEX_XERR')[0]
       .split('_ENDVERTEX_YERR')[0]
       .split('_ENDVERTEX_ZERR')[0] 
    for col in columns_found
})

for i in prefixes:
    print(f"- {i}")


particle = input('\nEnter the exact particle prefix you want to analyze: ').strip()

query = f"""
    SELECT 
        {particle}_M,
        {particle}_PT, 
        {particle}_PZ, 
        {particle}_ENDVERTEX_XERR, 
        {particle}_ENDVERTEX_YERR, 
        {particle}_ENDVERTEX_ZERR
    FROM read_parquet('{file_path}')
    WHERE {particle}_M IS NOT NULL  -- Ensure no corrupted rows from ROOT conversion
"""
df = con.execute(query).df()
print(f"Successfully loaded {len(df)} particle events.")


# -------------------------------------------------------------------------
# 2. Define Features and Target
# -------------------------------------------------------------------------
features = [
    f'{particle}_M',
    f'{particle}_PT', 
    f'{particle}_PZ', 
    f'{particle}_ENDVERTEX_XERR', 
    f'{particle}_ENDVERTEX_YERR', 
    f'{particle}_ENDVERTEX_ZERR'
]

query2 = f"""
    SELECT 
        SQRT(GREATEST(0, {particle}_PE^2 - ({particle}_PX^2 + {particle}_PY^2 + {particle}_PZ^2))) AS m_inv,
        {particle}_M as mass 
    FROM '{file_path}'
    WHERE {particle}_PX IS NOT NULL AND {particle}_PY IS NOT NULL AND {particle}_PZ IS NOT NULL AND {particle}_PE IS NOT NULL
"""
# Calculate the invariant mass and compare it with the given mass in the .root file to find the distortion

df2 = con.execute(query2).df()

mass_array = df2["m_inv"].to_numpy()

max_count = np.argmax(mass_array)

data = con.execute(query2).fetchnumpy()
counts, bin_edges = np.histogram(mass_array, bins=np.size(mass_array))
peak_bin_index = np.argmax(counts)
peak_mass_meV = (bin_edges[peak_bin_index] + bin_edges[peak_bin_index + 1]) / 2


df[f'{particle}_M'] = peak_mass_meV - df2["mass"]

X = df[features]
y = df[f'{particle}_M']

# Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, df.index, test_size=0.2, random_state=42
)

# -------------------------------------------------------------------------
# 3. Train XGBoost Regressor
# -------------------------------------------------------------------------
regressor = xgb.XGBRegressor(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

print("\nTraining XGBoost Regressor for Resolution Sharpening...")
regressor.fit(X_train, y_train)

# -------------------------------------------------------------------------
# 4. Apply Correction (Resolution Sharpening)
# -------------------------------------------------------------------------
predicted_distortion = regressor.predict(X_test)
original_mass_test = df.loc[idx_test, f'{particle}_M']

# Sharpening: Smeared Mass + Predicted Distortion
sharpened_mass_test = original_mass_test + predicted_distortion

# -------------------------------------------------------------------------
# 5. Evaluate Metrics
# -------------------------------------------------------------------------
sigma_before = np.std(original_mass_test)
sigma_after = np.std(sharpened_mass_test)
resolution_improvement = ((sigma_before - sigma_after) / sigma_before) * 100

print("\n--- Evaluation Results ---")
print(f"Original {particle} Mass Peak Sigma (Before): {sigma_before:.3f} MeV")
print(f"Sharpened {particle} Mass Peak Sigma (After):  {sigma_after:.3f} MeV")
print(f"Resolution Improvement: {resolution_improvement:.2f}%")
print(f"Regression Test RMSE: {np.sqrt(mean_squared_error(y_test, predicted_distortion)):.3f} MeV")