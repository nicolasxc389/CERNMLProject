# CERN ML Project: B Meson Decay Classification

> 🚀 Machine learning pipeline for classifying B meson decay events from real CERN collision data.
> Work in progress — code and results subject to change.

## Overview

This project applies machine learning to real particle physics data from CERN to classify **B± → J/ψ(→ μ⁺μ⁻)K±** decay events as signal or background. The pipeline combines physics-informed feature engineering with gradient boosting classifiers.

**Status:** Core modules complete (data processing, feature engineering). ML classification & refinements in progress.

---

## Data Source

Real CERN B Physics dataset from [CERN OpenData](https://opendata.cern.ch/):
- **Dataset 1:** [B±→J/ψ(→μ+μ−)K±CC Ntuples 4530 — 2017 Magnet Down](https://opendata.cern.ch/record/93949)
- **Dataset 2:** [B±→J/ψ(→μ+μ−)K±CC Ntuples 4530 — 2017 Magnet Up](https://opendata.cern.ch/record/93948)

The data contains B meson decays recorded by CERN detectors, with signal events (true decay) in the invariant mass peak and background events in the sidebands.

---

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

---

## Requirements

```
pandas>=1.3.0, uproot>=4.0.0, duckdb>=0.5.0, numpy>=1.21.0
matplotlib>=3.4.0, scikit-learn>=1.0.0, xgboost>=1.5.0
```

Install: `pip install -r requirements.txt`

---

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

---

## Physics Context

### Decay Channel: B± → J/ψ(→ μ⁺μ⁻)K±

This is a classic channel for CP-violation studies at CERN. Key features:

- The B meson (bottom quark) is essential for understanding matter-antimatter asymmetry
- Muons from J/ψ decay provide clean, detectable tracks in the detector
- Kaons complete the decay chain
- **Signal events** form a peak in the invariant mass spectrum near **5.28 GeV**
- **Background events** populate the sidebands

### Key Observable Quantities

- **Transverse Momentum (pT):** Momentum perpendicular to the beam axis
- **Pseudorapidity (η):** Angular measurement in detector-friendly coordinates
- **Invariant Mass:** Reconstructed rest mass from 4-vector algebra (E² - p²)

---

## Notes

- **Learning Project:** Focus is on understanding particle physics + ML workflows
- **Data Size:** Original ROOT files are large (~GB); download from CERN OpenData as needed
- **Python 3.8+** recommended
- **Status:** Work in progress; code and results are subject to change

---

## References

- [CERN OpenData Portal](https://opendata.cern.ch/)
- [ROOT Data Analysis Framework](https://root.cern/)
- [Uproot Documentation](https://uproot.readthedocs.io/)
- [DuckDB Documentation](https://duckdb.org/)
- [Particle Data Group (PDG)](https://pdg.lbl.gov/)
