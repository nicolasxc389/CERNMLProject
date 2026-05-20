#Imports all neccessary libraries for the code to run

import uproot
import tkinter as tk
from tkinter import filedialog

# This prompts the user to select a ROOT file
root = tk.Tk()
root.withdraw()

file_path = filedialog.askopenfilename(title="Select a ROOT file")

#This defines the input variable as the file path of the selected ROOT file
input = file_path

#This opens the ROOT file using uproot
file = uproot.open(input)

#This is accessing the tree
tree = file["Btree/DecayTree"]

#Converts specific branches of the trees into a dictionary of NumPy arrays
data = tree.arrays(["Bplus_ENDVERTEX_X"], library="np")

#Access the NumPy arrays
Bplus_ENDVERTEX_x = data["Bplus_ENDVERTEX_X"]

print(Bplus_ENDVERTEX_x)