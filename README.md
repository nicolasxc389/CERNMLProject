# CERN ML Project: B Meson Decay Classification

> ⚠️ **Work in Progress** — This project is under active development and serves as a learning journey in particle physics data analysis and machine learning. Results and code are subject to change.

## Overview

This project demonstrates an end-to-end machine learning pipeline for analyzing real particle physics data from CERN. The goal is to **classify B meson decay events** as signal or background using data from CERN's OpenData portal.

**Key Learning Objectives:**
- Extract and process real high-energy physics data (ROOT format)
- Engineer physics-informed features (momentum, pseudorapidity, invariant mass)
- Apply ML algorithms (XGBoost, scikit-learn) to particle physics classification
- Move from zero domain knowledge to defensible results

**Current Status:** Data processing and feature engineering modules complete. ML classification module in development.

---

## Data Source

This project uses **real CERN collision data** from the B Physics dataset:

- **Dataset 1:** [B±→J/ψ(→μ+μ−)K±CC Ntuples 4530 — 2017 Magnet Down](https://opendata.cern.ch/record/93949)
- **Dataset 2:** [B±→J/ψ(→μ+μ−)K±CC Ntuples 4530 — 2017 Magnet Up](https://opendata.cern.ch/record/93948)

The data contains B meson decays recorded by CERN detectors, with signal events (true decay) in the invariant mass peak and background events in the sidebands.

---

## Project Pipeline

```
parquet.py                    ← Load ROOT files, convert to Parquet
   ↓
momentum.py                   ← Analyze particle momentum distributions
   ↓
pseudorapidity.py             ← Calculate η, φ, pT features
   ↓
invariantmass.py              ← Reconstruct J/ψ and B meson masses
   ↓
boostedtrees.py               ← Train/evaluate ML classifier (IN PROGRESS)
```

### Module Descriptions

| Module | Purpose | Status |
|--------|---------|--------|
| `parquet.py` | Convert ROOT files to Parquet via tkinter file selector | ✅ Complete |
| `momentum.py` | Compute total momentum distributions with SQL queries | ✅ Complete |
| `pseudorapidity.py` | Calculate η, φ, pT with statistical analysis | ✅ Complete |
| `invariantmass.py` | Relativistic 4-vector algebra for mass reconstruction | ✅ Complete |
| `boostedtrees.py` | Gradient boosting classification & model evaluation | 🔨 In Progress |

---

## Requirements

```
pandas>=1.3.0           # Data manipulation and framing
uproot>=4.0.0          # Reading CERN ROOT files
duckdb>=0.5.0          # SQL-based querying of Parquet data
numpy>=1.21.0          # Numerical operations
matplotlib>=3.4.0      # Visualization and plotting
tkinter                 # GUI for file selection (usually included with Python)
scikit-learn>=1.0.0    # Machine learning algorithms
xgboost>=1.5.0         # Gradient boosting trees
```

### Installation

```bash
pip install -r requirements.txt
```

---

## Usage

### Step 1: Convert ROOT to Parquet
```bash
python parquet.py
# A file dialog will open. Select a ROOT file from CERN OpenData.
# Output: root_data_output.parquet
```

### Step 2: Analyze Momentum Distributions
```bash
python momentum.py
# Enter particle name when prompted (e.g., "muplus", "Kplus")
# Output: {particle}_momentum_distributions.png
```

### Step 3: Pseudorapidity & Angular Analysis
```bash
python pseudorapidity.py
# Enter particle name when prompted
# Output: {particle}_pseudorapidity_analysis.png + statistical summary
```

### Step 4: Invariant Mass Reconstruction
```bash
python invariantmass.py
# Enter particle prefixes when prompted for the B → J/ψ K decay chain
# Output: invariantmass.png (J/ψ and B mass peaks)
```

### Step 5: Machine Learning Classification *(In Progress)*
```bash
python boostedtrees.py
# Train/evaluate gradient boosting classifier on signal vs. background
```

---

## Physics Background

### B Meson Decay Chain
This project focuses on: **B± → J/ψ(→ μ⁺μ⁻)K±**

- The B meson (bottom quark) is a key particle in CP-violation studies
- Muons from J/ψ decay provide clean, detectable tracks
- Kaons complete the decay chain
- Signal events appear as a peak in the invariant mass spectrum near 5.28 GeV
- Background events populate the sidebands

### Key Physics Quantities
- **Transverse Momentum (pT):** Momentum perpendicular to beam axis
- **Pseudorapidity (η):** Angular measurement in detector-friendly coordinates
- **Invariant Mass:** Reconstructed mass from 4-vector algebra (E² - p²)

---

## Features & Roadmap

### ✅ Completed
- ROOT file processing and Parquet conversion
- Momentum distribution analysis
- Pseudorapidity and angular feature extraction
- Invariant mass reconstruction with relativistic kinematics
- SQL-based data querying with DuckDB

### 🔨 In Progress
- Complete gradient boosting classifier (boostedtrees.py)
- Train/test split and model evaluation
- ROC curves and classification metrics
- Overtraining analysis

### 📋 Future Enhancements
- Hyperparameter optimization (grid search, Bayesian optimization)
- Feature importance analysis
- Cross-validation and k-fold testing
- Visualizations of decision boundaries
- Documentation of results and findings

---

## Notes & Disclaimers

- **Work in Progress:** Code and results are subject to change as the project develops
- **Learning Project:** Emphasis is on understanding particle physics + ML workflows, not production optimization
- **Data Size:** Original ROOT files are large (~GB). Download from CERN OpenData as needed
- **Environment:** Tested on Python 3.8+. YMMK on different OS/Python versions

---

## Author

[@nicolasxc389](https://github.com/nicolasxc389)

---

## References

- [CERN OpenData Portal](https://opendata.cern.ch/)
- [ROOT Data Analysis Framework](https://root.cern/)
- [Uproot Documentation](https://uproot.readthedocs.io/)
- [DuckDB Documentation](https://duckdb.org/)
- [Particle Data Group (PDG)](https://pdg.lbl.gov/)
