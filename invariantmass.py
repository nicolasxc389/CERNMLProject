from cProfile import label
import math
import duckdb
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd


# Connect to the file
con = duckdb.connect()

# Let SQL do the filtering for columns containing 'P' or 'p'
query_X = """
    SELECT column_name 
    FROM (DESCRIBE SELECT * FROM 'root_data_output.parquet')
    WHERE column_name ILIKE '%_PX%'
"""
query_Y = """
    SELECT column_name 
    FROM (DESCRIBE SELECT * FROM 'root_data_output.parquet')
    WHERE column_name ILIKE '%_PY%'
"""
query_Z = """
    SELECT column_name 
    FROM (DESCRIBE SELECT * FROM 'root_data_output.parquet')
    WHERE column_name ILIKE '%_PZ%'
"""

query_E = """
    SELECT column_name
    FROM (DESCRIBE SELECT * FROM 'root_data_output.parquet')
    WHERE column_name ILIKE '%_PE%'
"""
# Fetch the results straight into a clean Python list
columns_with_momentum_X = con.execute(query_X).fetchall()
columns_with_momentum_Y = con.execute(query_Y).fetchall()
columns_with_momentum_Z = con.execute(query_Z).fetchall()
columns_with_energy = con.execute(query_E).fetchall()

column_list = [col[0] for col in columns_with_momentum_X + columns_with_momentum_Y + columns_with_momentum_Z + columns_with_energy]

print("Found momentum-related columns:")
print(column_list)
print("\n\nPlease input for what particle without _PX, _PY, and _PZ you would like to use for invariant mass calculation:")
particle = input()

# Configure the system to use matplotlib
matplotlib.use('Agg')

# Query to see the relevant columns for the chosen particle
query = f"""
    SELECT 
        SQRT(GREATEST(0, {particle}_PE^2 - ({particle}_PX^2 + {particle}_PY^2 + {particle}_PZ^2))) AS m_inv
    FROM 'root_data_output.parquet'
    WHERE {particle}_PX IS NOT NULL AND {particle}_PY IS NOT NULL AND {particle}_PZ IS NOT NULL AND {particle}_PE IS NOT NULL
"""
df = con.execute(query).df()

mass_array = df["m_inv"].to_numpy()

max_count = np.argmax(mass_array)

# Extract the data from the query into a NumPy structured array
try:
    data = con.execute(query).fetchnumpy()
    counts, bin_edges = np.histogram(mass_array, bins=np.size(mass_array))
    peak_bin_index = np.argmax(counts)
    peak_mass_meV = (bin_edges[peak_bin_index] + bin_edges[peak_bin_index + 1]) / 2
    
    # Set up a 1-panel plotting grid (1 row, 4 columns)
    fig, ax = plt.subplots(1, 1, figsize=(24, 5))
    bins = 101

    # Panel 1: Total Combined Momentum Magnitude (P)
    ax.hist(data['m_inv'], bins=bins, histtype='step', color='darkorange', lw=2)
    ax.set_title(f"${particle}$ Invariant Mass ($m$)")
    ax.set_xlabel("Invariant Mass [MeV/c²]")
    ax.set_ylabel("Counts")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(float(np.min(mass_array)), float(np.max(mass_array)))

    plt.tight_layout()
    plt.axvline(x=np.average(mass_array), color='red', linestyle='--', linewidth=1.5, label=f"Average Invariant Mass: {math.trunc(np.average(mass_array))} MeV/c²")
    plt.axvline(x=peak_mass_meV, color='blue', linestyle='--', linewidth=1.5, label=f"Peak Invariant Mass: {math.trunc(peak_mass_meV)} MeV/c²")
    plt.legend()
    plt.show()

    # Saves the file
    plt.savefig(f"{particle}_invariant_mass.png", dpi=300)

    print(f"The Average invariant mass for {particle}: {math.trunc(np.average(mass_array))} MeV/c²")
    print(f"The Peak invariant mass for {particle}: {math.trunc(peak_mass_meV)} MeV/c²")

except Exception as e:
    print(f"An error occurred while processing the data: {e}")
