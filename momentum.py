import duckdb
import matplotlib
import matplotlib.pyplot as plt
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
"""

columns_found = [col[0] for col in con.execute(query_discover).fetchall()]

print("\nFound the following available particle prefixes:")
# Using a set comprehension simplifies the syntax and removes duplicates automatically
prefixes = sorted({col.split('_PX')[0].split('_PY')[0].split('_PZ')[0] for col in columns_found})

for p in prefixes:
    print(f"  - {p}")


particle = input('\nEnter the exact particle prefix you want to analyze: ').strip()

# Configure the system to use matplotlib
matplotlib.use('Agg')

# Query to see the relevant columns for the chosen particle
query = f"""
    SELECT 
        SQRT({particle}_PX^2 + {particle}_PY^2 + {particle}_PZ^2) AS p_tot
    FROM '{file_path}'
    WHERE {particle}_PX IS NOT NULL
"""
# Extract the data from the query into a NumPy structured array
try:
    data = con.execute(query).fetchnumpy()

    # Set up a 1-panel plotting grid (1 row, 4 columns)
    fig, ax = plt.subplots(1, 1, figsize=(24, 5))
    bins = 100

    # Panel 1: Total Combined Momentum Magnitude (P)
    ax.hist(data['p_tot'] / 1000, bins=bins, histtype='step', color='darkorange', lw=2)
    ax.set_title(f"${particle}$ Total Momentum ($P$)")
    ax.set_xlabel("Total Momentum [GeV/c]")
    ax.set_ylabel("Counts")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0,500)

    plt.tight_layout()
    plt.show()

    # Saves the file
    plt.savefig(f"{particle}_momentum_distributions.png", dpi=300)

except Exception as e:
    print(f"An error occurred while processing the data:{e}")
