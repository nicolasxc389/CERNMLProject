# High Energy Particle Collision ML Project: B Meson Decay Classification

> Machine learning pipeline for classifying B± → J/ψ(→ μ⁺μ⁻)K± decay events from CERN OpenData.

## Overview

Compact pipeline to process CERN ROOT data, extract physics features, and classify B± → J/ψK± events as signal or background.

**Status:** Finished — core processing, feature engineering, and classification completed.

---

## Data Source

Real CERN B Physics dataset from CERN OpenData (2017 magnet up/down ntuples).

---

## Pipeline

parquet.py → momentum.py → pseudorapidity.py → invariantmass.py → boostedtree+neural.py, Xgboostregressor.py → angularSeparation.py

| Module | Purpose | Status |
|--------|---------|--------|
| `parquet.py` | Convert ROOT files to Parquet | ✅ |
| `momentum.py` | Momentum distributions | ✅ |
| `pseudorapidity.py` | η, φ, pT features | ✅ |
| `invariantmass.py` | 4-vector mass reconstruction | ✅ |
| `boostedtree.py` | Gradient boosting classifier | ✅ |
| `angularSeparation.py` | Angular analysis | ✅ |
| `resolutionSharpening.py` | Resolution regressor | 🔨 In Progress |

---

## Requirements

pandas, uproot, duckdb, numpy, matplotlib, scikit-learn, xgboost

Install: `pip install -r requirements.txt`

---

## Quick Start

```bash
python parquet.py
python momentum.py
python pseudorapidity.py
python invariantmass.py
python boostedtree.py
python angularSeparation.py
# resolutionSharpening.py is optional / in progress
```

---

## Physics Context

- Decay: B± → J/ψ(→ μ⁺μ⁻)K± — signal peak near ~5.28 GeV
- Key observables: transverse momentum (pT), pseudorapidity (η), invariant mass

---

## Notes

- Research/learning project focused on particle physics + ML workflows
- Data files are large; download from CERN OpenData as needed
- Python 3.8+ recommended

---

## References

- CERN OpenData, ROOT, Uproot, DuckDB, scikit-learn, XGBoost
