# CERN ML Project: B Meson Decay Classification

> 🚀 Machine learning pipeline for classifying B meson decay events from real CERN collision data.
> Work in progress — code and results subject to change.

## Overview

This project applies machine learning to real particle physics data from CERN to classify **B± → J/ψ(→ μ⁺μ⁻)K±** decay events as signal or background. The pipeline combines physics-informed feature engineering with gradient boosting classifiers.

**Status:** Core modules complete (data processing, feature engineering). ML classification & refinements in progress.

## Data Source

Real CERN B Physics dataset from [CERN OpenData](https://opendata.cern.ch/):
- [2017 Magnet Down](https://opendata.cern.ch/record/93949)
- [2017 Magnet Up](https://opendata.cern.ch/record/93948)

## Pipeline

```
parquet.py → momentum.py → pseudorapidity.py → invariantmass.py → boostedtree.py
                                                                  ↓
                                                    angularSeparation.py
                                                           ↓
                                                 resolutionSharpening.py (in progress)
```

| Module | Purpose | Status |
|--------|---------|--------|
| `parquet.py` | Convert ROOT files to Parquet | ✅ |
| `momentum.py` | Momentum distributions | ✅ |
| `pseudorapidity.py` | η, φ, pT feature extraction | ✅ |
| `invariantmass.py` | 4-vector mass reconstruction | ✅ |
| `boostedtree.py` | Gradient boosting classifier | ✅ Core / 🔨 Refinement |
| `angularSeparation.py` | Angular separation analysis | ✅ |
| `resolutionSharpening.py` | XGBoost resolution regressor | 🔨 In Progress |

## Requirements

```
pandas>=1.3.0, uproot>=4.0.0, duckdb>=0.5.0, numpy>=1.21.0
matplotlib>=3.4.0, scikit-learn>=1.0.0, xgboost>=1.5.0
```

Install: `pip install -r requirements.txt`

## Quick Start

```bash
# Convert ROOT file to Parquet
python parquet.py

# Extract features
python momentum.py          # Momentum distributions
python pseudorapidity.py    # η, φ, pT analysis
python invariantmass.py     # Mass reconstruction

# Classify signal vs. background
python boostedtree.py

# Analyze angular properties
python angularSeparation.py

# Sharpen resolution estimates (in progress)
python resolutionSharpening.py
```

## Physics Context

**B± → J/ψ(→ μ⁺μ⁻)K±** is a classic channel for CP-violation studies. Signal events form a peak in the invariant mass spectrum (~5.28 GeV); background populates the sidebands. Key observables: transverse momentum (pT), pseudorapidity (η), and reconstructed invariant masses.

## Notes

- **Learning Project:** Focus on understanding particle physics + ML workflows
- **Data Size:** Original ROOT files are large (~GB); download from CERN OpenData as needed
- **Python 3.8+** recommended
