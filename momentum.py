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
print("\n\nPlease input for what proton without _PX, _PY, and _PZ you would like to use for momentum calculation:")
particle = input()

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
