import os
import duckdb
import numpy as np
import matplotlib 
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog

#Select the parquet file
parquet = tk.Tk()
parquet.withdraw()
file_path = filedialog.askopenfilename(title="Select a Parquet file", filetypes =[("Parquet files", "*.parquet")])

con = duckdb.connect()

# Let SQL do the filtering for columns containing 'P' or 'p'
query_X = f"""
    SELECT column_name 
    FROM (DESCRIBE SELECT * FROM '{file_path}')
    WHERE column_name ILIKE '%_PX%'
"""
query_Y = f"""
    SELECT column_name 
    FROM (DESCRIBE SELECT * FROM '{file_path}')
    WHERE column_name ILIKE '%_PY%'
"""
query_Z = f"""
    SELECT column_name 
    FROM (DESCRIBE SELECT * FROM '{file_path}')
    WHERE column_name ILIKE '%_PZ%'
"""
# Fetch the results straight into a clean Python list
columns_with_momentum_X = con.execute(query_X).fetchall()
columns_with_momentum_Y = con.execute(query_Y).fetchall()
columns_with_momentum_Z = con.execute(query_Z).fetchall()

column_list = [col[0] for col in columns_with_momentum_X + columns_with_momentum_Y + columns_with_momentum_Z]

print("Found momentum-related columns:")
print(column_list)
print("\nEnter particle name (without _PX, _PY, _PZ):")
particle = input()

# Ensure that matplotlib can execute in a headless environment (like a server without a display)
matplotlib.use('Agg')

# Query to calculate pseudorapidity
query = f"""
    SELECT 
        SQRT({particle}_PX^2 + {particle}_PY^2 + {particle}_PZ^2) AS p_tot,
        ATANH({particle}_PZ / SQRT({particle}_PX^2 + {particle}_PY^2 + {particle}_PZ^2)) AS eta,
        ATAN2({particle}_PY, {particle}_PX) AS phi,
        SQRT({particle}_PX^2 + {particle}_PY^2) AS pt
    FROM '{file_path}'
    WHERE {particle}_PX IS NOT NULL
        AND {particle}_PY IS NOT NULL
        AND {particle}_PZ IS NOT NULL
        AND ABS({particle}_PZ / SQRT({particle}_PX^2 + {particle}_PY^2 + {particle}_PZ^2)) < 1.0
"""

try:
    # Fetch as pandas DataFrame instead (more robust)
    df = con.execute(query).df()
    
    print(f"\nTotal particles: {len(df)}")
    
    # Remove NaN values from eta
    df_clean = df.dropna(subset=['eta'])
    print(f"Valid particles (finite η): {len(df_clean)}")
    
    # Convert to numpy arrays for easier plotting
    eta = df_clean['eta'].values
    pt = df_clean['pt'].values / 1000  # Convert to GeV
    phi = df_clean['phi'].values
    p_tot = df_clean['p_tot'].values / 1000  # Convert to GeV
    
    # Set up a 2x2 plotting grid
    fig, ax = plt.subplots(2, 2, figsize=(14, 10))
    bins = 100
    
    # Panel 1: Pseudorapidity Distribution
    ax[0, 0].hist(eta, bins=bins, histtype='step', color='blue', lw=2, edgecolor='blue')
    ax[0, 0].set_title(f"${particle}$ Pseudorapidity (η)", fontsize=12, fontweight='bold')
    ax[0, 0].set_xlabel("Pseudorapidity (η)")
    ax[0, 0].set_ylabel("Counts")
    ax[0, 0].grid(True, alpha=0.3)
    ax[0, 0].axvline(np.mean(eta), color='red', linestyle='--', linewidth=2, label=f"Mean: {np.mean(eta):.3f}")
    ax[0, 0].set_xlim(0, 7)
    ax[0, 0].legend()
    
    # Panel 2: Transverse Momentum
    ax[0, 1].hist(pt, bins=bins, histtype='step', color='green', lw=2, edgecolor='green')
    ax[0, 1].set_title(f"${particle}$ Transverse Momentum ($p_T$)", fontsize=12, fontweight='bold')
    ax[0, 1].set_xlabel("Transverse Momentum [GeV/c]")
    ax[0, 1].set_ylabel("Counts")
    ax[0, 1].grid(True, alpha=0.3)
    
    # Panel 3: Azimuthal Angle (φ)
    ax[1, 0].hist(phi, bins=bins, histtype='step', color='purple', lw=2, edgecolor='purple')
    ax[1, 0].set_title(f"${particle}$ Azimuthal Angle (φ)", fontsize=12, fontweight='bold')
    ax[1, 0].set_xlabel("Azimuthal Angle (radians)")
    ax[1, 0].set_ylabel("Counts")
    ax[1, 0].grid(True, alpha=0.3)
    
    # Panel 4: η vs pT scatter plot
    scatter = ax[1, 1].scatter(eta, pt, alpha=0.3, s=5, c='darkblue')
    ax[1, 1].set_title(f"${particle}$ η vs $p_T$ Correlation", fontsize=12, fontweight='bold')
    ax[1, 1].set_xlabel("Pseudorapidity (η)")
    ax[1, 1].set_ylabel("Transverse Momentum [GeV/c]")
    ax[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{particle}_pseudorapidity_analysis.png", dpi=300)
    plt.show()
    
    # ===== PRINT STATISTICS =====
    print("\n" + "="*60)
    print(f"PSEUDORAPIDITY ANALYSIS: {particle}")
    print("="*60)
    print(f"Mean η:              {np.mean(eta):.4f}")
    print(f"Std Dev:             {np.std(eta):.4f}")
    print(f"Median η:            {np.median(eta):.4f}")
    print(f"Min η:               {np.min(eta):.4f}")
    print(f"Max η:               {np.max(eta):.4f}")
    print("\nPercentiles:")
    print(f"  1st percentile:    {np.percentile(eta, 1):.4f}")
    print(f"  25th percentile:   {np.percentile(eta, 25):.4f}")
    print(f"  75th percentile:   {np.percentile(eta, 75):.4f}")
    print(f"  99th percentile:   {np.percentile(eta, 99):.4f}")
    
    print(f"\nTransverse Momentum (pT):")
    print(f"  Mean pT:           {np.mean(pt):.4f} GeV/c")
    print(f"  Std Dev:           {np.std(pt):.4f} GeV/c")
    print(f"  Max pT:            {np.max(pt):.4f} GeV/c")
    
    print(f"\nTotal Momentum (p):")
    print(f"  Mean p:            {np.mean(p_tot):.4f} GeV/c")
    print(f"  Std Dev:           {np.std(p_tot):.4f} GeV/c")
    print(f"  Max p:             {np.max(p_tot):.4f} GeV/c")
    print("="*60 + "\n")
    
except Exception as e:
    print(f"An error occurred: {e}")
    import traceback
    traceback.print_exc()
