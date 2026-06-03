import math
import duckdb
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog

#Select the parquet file
parquet = tk.Tk()
parquet.withdraw()
file_path = filedialog.askopenfilename(title="Select a Parquet file", filetypes =[("Parquet files", "*.parquet")])

# Connect to the file
con = duckdb.connect()

query_discover = f"""
    SELECT column_name 
    FROM (DESCRIBE SELECT * FROM '{file_path}') 
    WHERE column_name ILIKE '%_PX%' 
       OR column_name ILIKE '%_PY%' 
       OR column_name ILIKE '%_PZ%' 
       OR column_name ILIKE '%_PE%'
"""

columns_found = [col[0] for col in con.execute(query_discover).fetchall()]

print("\nFound the following available particle prefixes:")
# Using a set comprehension simplifies the syntax and removes duplicates automatically
prefixes = sorted({col.split('_PX')[0].split('_PY')[0].split('_PZ')[0].split('_PE')[0] for col in columns_found})

for p in prefixes:
    print(f"  - {p}")


particle = input('\nEnter the exact particle prefix you want to analyze: ').strip()


# Configure the system to use matplotlib
matplotlib.use('Agg')

# Query to see the relevant columns for the chosen particle
query = f"""
    SELECT 
        SQRT(GREATEST(0, {particle}_PE^2 - ({particle}_PX^2 + {particle}_PY^2 + {particle}_PZ^2))) AS m_inv
    FROM '{file_path}'
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

    #Save the DF to a parquet file to send to the boosted decision tree for classification
    output_filename = f"{particle}-invariantmass.parquet"
    df.to_parquet(output_filename, compression='snappy')


    print(f"The Average invariant mass for {particle}: {math.trunc(np.average(mass_array))} MeV/c²")
    print(f"The Peak invariant mass for {particle}: {math.trunc(peak_mass_meV)} MeV/c²")



except Exception as e:
    print(f"An error occurred while processing the data: {e}")
