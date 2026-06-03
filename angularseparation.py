import duckdb
import matplotlib
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import pandas as pd
import numpy as np

#Select the parquet file
parquet = tk.Tk()
parquet.withdraw()
file_path = filedialog.askopenfilename(title="Select a Parquet file", filetypes =[("Parquet files", "*.parquet")])

# Connect to the file
con = duckdb.connect()

query_discover = f"SELECT column_name FROM (DESCRIBE SELECT * FROM '{file_path}') WHERE column_name ILIKE '%_PX%'"
columns_found = [col[0] for col in con.execute(query_discover).fetchall()]

print("\nFound the following available particle prefixes:")
prefixes = sorted(list(set([col.split('_PX')[0] for col in columns_found])))
for p in prefixes:
    print(f"  - {p}")

first_particle = input('\nEnter the first particle prefix you want to analyze: ').strip()
second_particle = input('Enter the second particle prefix you want to analyze: ').strip()

# ======================================================
# START PSEUDORAPIDITY AND AZIMUTHAL ANGLE CALCULATIONS
# ======================================================

first_particle_query = f"""
    SELECT 
        ATANH({first_particle}_PZ / SQRT({first_particle}_PX^2 + {first_particle}_PY^2 + {first_particle}_PZ^2)) AS eta1,
        ATAN2({first_particle}_PY, {first_particle}_PX) AS phi1
    FROM '{file_path}'
    WHERE {first_particle}_PX IS NOT NULL
        AND {first_particle}_PY IS NOT NULL
        AND {first_particle}_PZ IS NOT NULL
        AND ABS({first_particle}_PZ / SQRT({first_particle}_PX^2 + {first_particle}_PY^2 + {first_particle}_PZ^2)) < 1.0
"""
second_particle_query = f"""
    SELECT
        ATANH({second_particle}_PZ / SQRT({second_particle}_PX^2 + {second_particle}_PY^2 + {second_particle}_PZ^2)) AS eta2,
        ATAN2({second_particle}_PY, {second_particle}_PX) AS phi2
    FROM '{file_path}'
    WHERE {second_particle}_PX IS NOT NULL
        AND {second_particle}_PY IS NOT NULL
        AND {second_particle}_PZ IS NOT NULL
        AND ABS({second_particle}_PZ / SQRT({second_particle}_PX^2 + {second_particle}_PY^2 + {second_particle}_PZ^2)) < 1.0    
"""

df_1 = con.execute(first_particle_query).df()
df_2 = con.execute(second_particle_query).df()

print(f"\nTotal particles: {len(df_1)}")
print(f"\nTotal particles: {len(df_2)}")

# Calculate differences
d_eta = df_1['eta1'].values - df_2['eta2'].values  
d_phi = df_1['phi1'].values - df_2['phi2'].values

# Normalize d_phi to [-pi, pi]
d_phi = np.arctan2(np.sin(d_phi), np.cos(d_phi))

# Calculate corrected angular separation
angular_separation = np.sqrt(d_eta**2 + d_phi**2)


counts, bin_edges = np.histogram(angular_separation, bins=np.size(angular_separation))
peak_bin_index = np.argmax(counts)
peak_angular_separation = (bin_edges[peak_bin_index] + bin_edges[peak_bin_index + 1]) / 2

# Ensure that matplotlib can execute in a headless environment (like a server without a display)
matplotlib.use('Agg')

fig, ax = plt.subplots(figsize=(14, 10))
bins = 100

# Panel: Angular Separation (ΔR) between the two particles
ax.hist(angular_separation, bins=bins, histtype='step', color='blue', lw=2, edgecolor='blue')
ax.set_title(f"Angular Separation (ΔR) between {first_particle} and {second_particle}", fontsize=12, fontweight='bold')
ax.set_xlabel("Angular Separation (ΔR)")
ax.set_ylabel("Counts")
ax.grid(True, alpha=0.3)
ax.axvline(np.mean(angular_separation), color='red', linestyle='--', linewidth=2, label=f"Mean: {np.mean(angular_separation):.3f}")
ax.axvline(x=peak_angular_separation, color='blue', linestyle='--', linewidth=1.5, label=f"Peak Angular Separation: {peak_angular_separation:.3f}")
ax.legend()

plt.tight_layout()
plt.tight_layout()
plt.savefig(f"{first_particle}_{second_particle}_angular_separation.png", dpi=300)
plt.show()

print(f"\nMean Angular Separation (ΔR) between {first_particle} and {second_particle}: {np.mean(angular_separation):.3f}")