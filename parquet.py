import uproot
import tkinter as tk
from tkinter import filedialog
import pandas as pd  

# Select the ROOT file
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename(title="Select a ROOT file", filetypes =[("ROOT files", "*.root")])

# Open and extract all data directly into a Pandas DataFrame
with uproot.open(file_path) as file:
    tree = file["Btree/DecayTree"]
    
    # Leaving the first argument empty makes uproot read ALL branches
    df = tree.arrays(library="pd")

# Save to Parquet format instantly
output_filename = f"{file_path}.parquet"
df.to_parquet(output_filename, compression="snappy")

print(f"Successfully saved {len(df)} rows and {len(df.columns)} columns to {output_filename}")
