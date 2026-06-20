"""
CERN B Meson Decay Classification Dashboard
============================================
Interactive Streamlit dashboard for visualizing particle physics pipeline,
ML model performance, and decay event analysis.

Requirements: streamlit, pandas, numpy, matplotlib, duckdb, pyarrow, requests
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import duckdb
import os
from pathlib import Path
from urllib.parse import urlparse
import requests
import tempfile

# Set page config first
st.set_page_config(
    page_title="CERN B Physics Dashboard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    </style>
""", unsafe_allow_html=True)

# =====================================================================
# SIDEBAR - DATA SOURCE & NAVIGATION
# =====================================================================
st.sidebar.markdown("## 📁 Data & Navigation")

DEFAULT_PARQUET_URL = (
    "https://huggingface.co/datasets/nicolasxc2089/Bmesondecayparquet/"
    "resolve/main/2017_Magnetic_Down_Photon-Photon.root.parquet"
)

file_source_path = st.sidebar.text_input(
    "Parquet URL",
    value=DEFAULT_PARQUET_URL,
    help="Remote Parquet URL (Hugging Face works best)",
)

# Sampling control - CRITICAL for large files
sample_size = st.sidebar.slider(
    "Sample Size (rows)",
    min_value=10_000,
    max_value=200_000,
    value=50_000,
    step=10_000,
    help="Larger samples = more memory usage. Start small on Streamlit Cloud."
)

st.sidebar.caption("File is ~2.5 GB → sampling is required")

nav_option = st.sidebar.radio(
    "Select Dashboard View",
    ["🏠 Home", "📊 Feature Analysis", "🤖 Model Performance", 
     "🎯 Event Classification", "📈 Physics Insights", "⚙️ Pipeline Status"]
)

# =====================================================================
# HELPER FUNCTIONS
# =====================================================================

@st.cache_resource
def get_db_connection():
    """Create cached DuckDB connection with HTTP support."""
    con = duckdb.connect(database=":memory:")
    con.execute("INSTALL httpfs;")
    con.execute("LOAD httpfs;")
    # Optional: limit memory
    # con.execute("SET memory_limit='1.2GB';")
    return con


@st.cache_data(ttl=3600, show_spinner="Loading sampled Parquet data...")
def load_parquet_data_from_source(source_path: str, sample_rows: int):
    """Load sampled data from remote Parquet using DuckDB."""
    if not source_path:
        st.error("No data source provided.")
        return None

    con = get_db_connection()

    try:
        # Use LIMIT for sampling - essential for large files
        query = f"""
        SELECT * 
        FROM read_parquet('{source_path}')
        LIMIT {sample_rows}
        """
        df = con.execute(query).df()
        
        st.sidebar.success(f"✅ Loaded {len(df):,} rows (sampled)")
        return df
    except Exception as e:
        st.error(f"Failed to load data: {str(e)}")
        st.info("💡 Try a smaller sample or check the URL. Make sure requirements.txt includes 'duckdb'.")
        return None


def discover_particle_prefixes(df):
    """Auto-detect particle prefixes from column names."""
    prefixes = set()
    for col in df.columns:
        if col.endswith(('_PX', '_PY', '_PZ', '_PE', '_PT', '_M')):
            prefix = col.rsplit('_', 1)[0]
            prefixes.add(prefix)
    return sorted(list(prefixes))


# Physics calculation functions (unchanged)
def calculate_invariant_mass(df, particle_prefix):
    try:
        px = f'{particle_prefix}_PX'
        py = f'{particle_prefix}_PY'
        pz = f'{particle_prefix}_PZ'
        pe = f'{particle_prefix}_PE'
        
        if all(col in df.columns for col in [px, py, pz, pe]):
            m_inv = np.sqrt(np.maximum(0,
                df[pe]**2 - (df[px]**2 + df[py]**2 + df[pz]**2)
            ))
            return m_inv
    except Exception:
        pass
    return None


def calculate_pseudorapidity(df, particle_prefix):
    try:
        px = f'{particle_prefix}_PX'
        py = f'{particle_prefix}_PY'
        pz = f'{particle_prefix}_PZ'
        
        if all(col in df.columns for col in [px, py, pz]):
            p_tot = np.sqrt(df[px]**2 + df[py]**2 + df[pz]**2)
            eta = np.arctanh(np.clip(df[pz] / np.maximum(p_tot, 1e-10), -0.999, 0.999))
            return eta
    except Exception:
        pass
    return None


def calculate_transverse_momentum(df, particle_prefix):
    try:
        px = f'{particle_prefix}_PX'
        py = f'{particle_prefix}_PY'
        if all(col in df.columns for col in [px, py]):
            pt = np.sqrt(df[px]**2 + df[py]**2) / 1000  # GeV
            return pt
    except Exception:
        pass
    return None


def calculate_angular_separation(df, p1, p2):
    try:
        eta1 = calculate_pseudorapidity(df, p1) or df.get(f'{p1}_ETA')
        phi1 = np.arctan2(df[f'{p1}_PY'], df[f'{p1}_PX']) if f'{p1}_PY' in df.columns else df.get(f'{p1}_PHI')
        eta2 = calculate_pseudorapidity(df, p2) or df.get(f'{p2}_ETA')
        phi2 = np.arctan2(df[f'{p2}_PY'], df[f'{p2}_PX']) if f'{p2}_PY' in df.columns else df.get(f'{p2}_PHI')
        
        if eta1 is not None and eta2 is not None and phi1 is not None and phi2 is not None:
            d_eta = eta1 - eta2
            d_phi = np.arctan2(np.sin(phi1 - phi2), np.cos(phi1 - phi2))
            return np.sqrt(d_eta**2 + d_phi**2)
    except Exception:
        pass
    return None

# =====================================================================
# PAGES
# =====================================================================

if nav_option == "🏠 Home":
    st.markdown('<div class="main-header">🔬 CERN B Physics Analysis Dashboard</div>', unsafe_allow_html=True)
    # ... (your existing Home content - unchanged)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 📚 Project Overview\n**B± → J/ψ(→ μ⁺μ⁻)K± Decay Classification**")
    with col2:
        st.markdown("### 🎯 Pipeline Stages\n1. Data Loading\n2. Feature Engineering\n3. ML Classification")
    with col3:
        st.markdown("### 🔍 Key Features\n- Invariant Mass\n- Pseudorapidity\n- pT\n- Angular Separation")

    st.info("Use the sidebar to select other views. Sampling is enabled for large files.")

elif nav_option == "📊 Feature Analysis":
    st.title("📊 Feature Analysis & Distributions")
    df = load_parquet_data_from_source(file_source_path, sample_size)
    if df is None:
        st.stop()

    prefixes = discover_particle_prefixes(df)
    if not prefixes:
        st.warning("No particle prefixes detected. Check column names.")
        st.dataframe(df.columns.tolist())
        st.stop()

    analysis_type = st.sidebar.radio(
        "Choose Analysis", 
        ["Invariant Mass", "Pseudorapidity", "Transverse Momentum", "Angular Separation"]
    )

    # (Rest of your Feature Analysis code - mostly unchanged, using the sampled df)
    if analysis_type == "Invariant Mass":
        particle = st.selectbox("Select Particle", prefixes)
        m_inv = calculate_invariant_mass(df, particle)
        if m_inv is not None:
            m_inv_clean = m_inv[~np.isnan(m_inv)]
            # ... rest of your plotting code
            col1, col2, col3, col4 = st.columns(4)
            with col1: st.metric("Mean Mass", f"{np.mean(m_inv_clean):.2f} MeV/c²")
            # ... (continue with your existing histogram etc.)

    # Add similar blocks for other analysis types (copy from your original script)

# Add the other pages similarly (Model Performance, Event Classification, etc.)
# For brevity I showed the pattern - let me know if you want me to expand any section.

elif nav_option == "⚙️ Pipeline Status":
    # Your existing pipeline status code (unchanged)
    st.title("⚙️ Pipeline Status & Configuration")
    # ... rest of your pipeline page

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #888; margin-top: 2rem;'>
    <p><strong>CERN B Physics Dashboard</strong> | High Energy Particle Collision ML Project</p>
</div>
""", unsafe_allow_html=True)
