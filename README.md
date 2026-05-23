# CERNMLProject

   This is another work in progress project I am doing. So far, the outcome of this project is to move from zero domain knowledge to a defensible result, either a measurement, an improved classification, or even just showing what data tells us. This involves extracting       real data from mutliple datasets, such as Proton-Proton collisions produced from CERN's CMS and ATLAS 13 TeV jets, and using ML techniques/algorithms to analyze such data to produce a result. Again, at the time of this writing, this is a work in progress and not reflective of my actual abilities.


The required python libraries are:
   | Library | Purpose |
   |---------|---------|
   | **pandas** | Data manipulation and analysis |
   | **uproot** | Reading ROOT files (CERN data format) |
   | **duckdb** | SQL-based to read the .parquet data |
   | **numpy** | Numerical data inside the .parquet data |
   | **matplotlib** | Data graphing |
   | **tkinter** | File selection GUI |

**Pipeline**

     parquet.py
      │
      └──➔ momentum.py
      │
      └──➔ pseudorapidity.py


**Resources**

   https://opendata.cern.ch/record/93949
   https://opendata.cern.ch/record/93948
      
**Data used**

   - B±→J/ψ(→μ+μ−)K±CC Ntuples 4530 -- 2017 Magnet Down
   
   - B±→J/ψ(→μ+μ−)K±CC Ntuples 4530 -- 2017 Magnet Up

**Author**: 
[@nicolasxc389]

(https://github.com/nicolasxc389)
   
