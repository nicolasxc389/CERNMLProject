import streamlit as st
import pandas as pd
import numpy as np
import duckdb
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
import math

# Configure Streamlit page
st.set_page_config(
    page_title="B→J/ψK± Decay Analysis Dashboard",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2em;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .physics-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 30px;
        border-radius: 10px;
        margin-bottom: 30px;
    }
    </style>
    """, unsafe_allow_html=True)

matplotlib.use('Agg')

# Initialize session state
if 'parquet_file' not in st.session_state:
    st.session_state.parquet_file = None
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================
st.sidebar.markdown("## 📊 Navigation")
page = st.sidebar.radio(
    "Select Analysis Module:",
    ["🏠 Home & Overview", 
     "📈 Momentum Analysis", 
     "⚛️ Invariant Mass",
     "🎯 Pseudorapidity (η,φ,pT)",
     "🔄 Angular Separation",
     "🤖 ML Models"]
)

# ============================================================================
# DATA LOADING
# ============================================================================
st.sidebar.markdown("---")
st.sidebar.markdown("## 📁 Data Input")
uploaded_file = st.sidebar.file_uploader("Upload Parquet File", type="parquet")

if uploaded_file is not None:
    st.session_state.parquet_file = uploaded_file
    st.session_state.data_loaded = True
    file_path = uploaded_file.name

# Function to get available particles
@st.cache_data
def get_available_particles(file_path):
    try:
        con = duckdb.connect()
        query_discover = f"""
            SELECT column_name 
            FROM (DESCRIBE SELECT * FROM '{file_path}') 
            WHERE column_name ILIKE '%_PX%' 
               OR column_name ILIKE '%_PY%' 
               OR column_name ILIKE '%_PZ%'
        """
        columns_found = [col[0] for col in con.execute(query_discover).fetchall()]
        prefixes = sorted({col.split('_PX')[0].split('_PY')[0].split('_PZ')[0] for col in columns_found})
        return prefixes
    except Exception as e:
        st.error(f"Error discovering particles: {e}")
        return []

# ============================================================================
# PAGE: HOME & OVERVIEW
# ============================================================================
if page == "🏠 Home & Overview":
    st.markdown("""
    <div class="physics-header">
        <h1>⚛️ B Meson Decay Analysis Dashboard</h1>
        <h3>B± → J/ψ(→ μ⁺μ⁻)K± Particle Physics Visualization</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # README content
    st.markdown("""
    # High Energy Particle Collision ML Project: B Meson Decay Classification
    
    > 🚀 Machine learning pipeline for classifying B meson decay events from real CERN collision data.
    
    ## Overview
    
    This dashboard applies machine learning to real particle physics data from CERN to classify **B± → J/ψ(→ μ⁺μ⁻)K±** decay events 
    as signal or background. The pipeline combines physics-informed feature engineering with gradient boosting algorithms.
    
    **Status:** Core modules complete (data processing, feature engineering). ML classification & refinements in progress.
    
    ---
    
    ## Data Source
    
    Real CERN B Physics dataset from [CERN OpenData](https://opendata.cern.ch/):
    - **Dataset 1:** [B±→J/ψ(→μ+μ−)K±CC Ntuples 4530 — 2017 Magnet Down](https://opendata.cern.ch/record/93949)
    - **Dataset 2:** [B±→J/ψ(→μ+μ−)K±CC Ntuples 4530 — 2017 Magnet Up](https://opendata.cern.ch/record/93948)
    
    The data contains B meson decays recorded by CERN detectors, with signal events (true decay) in the invariant mass peak 
    and background events in the sidebands.
    
    ---
    
    ## Pipeline Architecture
    
    ```
    parquet.py → momentum.py → pseudorapidity.py → invariantmass.py → boostedtree.py
                                                                          ↓
                                                            angularSeparation.py
    ```
    
    | Module | Purpose | Status |
    |--------|---------|--------|
    | `parquet.py` | Convert ROOT files to Parquet | ✅ |
    | `momentum.py` | Momentum distributions | ✅ |
    | `pseudorapidity.py` | η, φ, pT feature extraction | ✅ |
    | `invariantmass.py` | 4-vector mass reconstruction | ✅ |
    | `xgboostregressor.py` | XGBoost classifier | ✅ |
    | `boostedtree+neural.py` | Gradient boosting + neural network | ✅ |
    | `angularSeparation.py` | Angular separation analysis | ✅ |
    
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
    - **Angular Separation (ΔR):** Distance between particles in η-φ space
    
    ---
    
    ## How to Use This Dashboard
    
    1. **Upload Data:** Use the sidebar to upload a parquet file
    2. **Select Analysis:** Choose from the navigation menu
    3. **Choose Particle:** Select which particle to analyze (Bplus, J_psi_1S, Kplus, mu_minus, mu_plus)
    4. **Visualize:** View interactive plots and statistics
    
    ---
    
    ## Requirements
    
    ```
    pandas>=1.3.0, uproot>=4.0.0, duckdb>=0.5.0, numpy>=1.21.0
    matplotlib>=3.4.0, scikit-learn>=1.0.0, xgboost>=1.5.0, streamlit>=1.0.0
    ```
    
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
    """)
    
    # Quick stats cards
    st.markdown("---")
    st.subheader("📊 Key Particles in Analysis")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.info("**B⁺**\\nParent particle")
    with col2:
        st.info("**J/ψ(1S)**\\nIntermediate state")
    with col3:
        st.info("**μ⁻**\\nLepton track")
    with col4:
        st.info("**μ⁺**\\nLepton track")
    with col5:
        st.info("**K⁺**\\nHadron")

# ============================================================================
# PAGE: MOMENTUM ANALYSIS
# ============================================================================
elif page == "📈 Momentum Analysis":
    st.header("📈 Momentum Analysis")
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please upload a parquet file in the sidebar first.")
    else:
        con = duckdb.connect()
        particles = get_available_particles(file_path)
        
        st.markdown("Calculate the **total momentum magnitude** (P) for selected particles:")
        
        particle = st.selectbox("Select Particle:", particles, key="momentum_particle")
        
        if st.button("Calculate Momentum Distribution", key="momentum_btn"):
            with st.spinner("Calculating momentum distribution..."):
                try:
                    query = f"""
                        SELECT 
                            SQRT({particle}_PX^2 + {particle}_PY^2 + {particle}_PZ^2) AS p_tot
                        FROM '{file_path}'
                        WHERE {particle}_PX IS NOT NULL
                    """
                    data = con.execute(query).df()
                    momentum_array = data['p_tot'].to_numpy() / 1000
                    
                    counts, bin_edges = np.histogram(momentum_array, bins=100)
                    peak_bin_index = np.argmax(counts)
                    peak_momentum = (bin_edges[peak_bin_index] + bin_edges[peak_bin_index + 1]) / 2
                    
                    fig, ax = plt.subplots(figsize=(12, 6))
                    ax.hist(momentum_array, bins=100, histtype='step', color='darkorange', lw=2.5)
                    ax.axvline(np.mean(momentum_array), color='red', linestyle='--', linewidth=2, 
                              label=f"Mean: {np.mean(momentum_array):.3f} MeV/c")
                    ax.axvline(peak_momentum, color='blue', linestyle='--', linewidth=2, 
                              label=f"Peak: {peak_momentum:.3f} MeV/c")
                    ax.set_title(f"{particle} Total Momentum Distribution", fontsize=14, fontweight='bold')
                    ax.set_xlabel("Total Momentum [MeV/c]", fontsize=12)
                    ax.set_ylabel("Counts", fontsize=12)
                    ax.grid(True, alpha=0.3)
                    ax.legend(fontsize=11)
                    st.pyplot(fig)
                    
                    # Statistics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Mean Momentum", f"{np.mean(momentum_array):.3f} MeV/c")
                    with col2:
                        st.metric("Peak Momentum", f"{peak_momentum:.3f} MeV/c")
                    with col3:
                        st.metric("Std Dev", f"{np.std(momentum_array):.3f} MeV/c")
                    with col4:
                        st.metric("Max Momentum", f"{np.max(momentum_array):.3f} MeV/c")
                    
                except Exception as e:
                    st.error(f"❌ Error: {e}")

# ============================================================================
# PAGE: INVARIANT MASS
# ============================================================================
elif page == "⚛️ Invariant Mass":
    st.header("⚛️ Invariant Mass Analysis")
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please upload a parquet file in the sidebar first.")
    else:
        con = duckdb.connect()
        particles = get_available_particles(file_path)
        
        st.markdown("Reconstruct the **invariant mass** using 4-vector algebra: **m = √(E² - p²)**")
        
        particle = st.selectbox("Select Particle:", particles, key="mass_particle")
        
        if st.button("Calculate Invariant Mass", key="mass_btn"):
            with st.spinner("Calculating invariant mass..."):
                try:
                    query = f"""
                        SELECT 
                            SQRT(GREATEST(0, {particle}_PE^2 - ({particle}_PX^2 + {particle}_PY^2 + {particle}_PZ^2))) AS m_inv
                        FROM '{file_path}'
                        WHERE {particle}_PX IS NOT NULL 
                            AND {particle}_PY IS NOT NULL 
                            AND {particle}_PZ IS NOT NULL 
                            AND {particle}_PE IS NOT NULL
                    """
                    data = con.execute(query).df()
                    mass_array = data['m_inv'].to_numpy()
                    
                    counts, bin_edges = np.histogram(mass_array, bins=101)
                    peak_bin_index = np.argmax(counts)
                    peak_mass = (bin_edges[peak_bin_index] + bin_edges[peak_bin_index + 1]) / 2
                    
                    fig, ax = plt.subplots(figsize=(12, 6))
                    ax.hist(mass_array, bins=101, histtype='step', color='darkorange', lw=2.5)
                    ax.axvline(np.mean(mass_array), color='red', linestyle='--', linewidth=2, 
                              label=f"Mean: {np.mean(mass_array):.1f} MeV/c²")
                    ax.axvline(peak_mass, color='blue', linestyle='--', linewidth=2, 
                              label=f"Peak: {peak_mass:.1f} MeV/c²")
                    ax.set_title(f"{particle} Invariant Mass", fontsize=14, fontweight='bold')
                    ax.set_xlabel("Invariant Mass [MeV/c²]", fontsize=12)
                    ax.set_ylabel("Counts", fontsize=12)
                    ax.grid(True, alpha=0.3)
                    ax.legend(fontsize=11)
                    st.pyplot(fig)
                    
                    # Statistics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Mean Mass", f"{np.mean(mass_array):.1f} MeV/c²")
                    with col2:
                        st.metric("Peak Mass", f"{math.trunc(peak_mass)} MeV/c²")
                    with col3:
                        st.metric("Std Dev", f"{np.std(mass_array):.1f} MeV/c²")
                    with col4:
                        st.metric("Max Mass", f"{np.max(mass_array):.1f} MeV/c²")
                    
                except Exception as e:
                    st.error(f"❌ Error: {e}")

# ============================================================================
# PAGE: PSEUDORAPIDITY ANALYSIS
# ============================================================================
elif page == "🎯 Pseudorapidity (η,φ,pT)":
    st.header("🎯 Pseudorapidity & Angular Analysis")
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please upload a parquet file in the sidebar first.")
    else:
        con = duckdb.connect()
        particles = get_available_particles(file_path)
        
        st.markdown("""
        Analyze pseudorapidity (η), azimuthal angle (φ), and transverse momentum (pT):
        - **η (Pseudorapidity):** log(tan(θ/2)) where θ is the polar angle
        - **φ (Azimuthal Angle):** Angle in the transverse plane
        - **pT (Transverse Momentum):** √(px² + py²)
        """)
        
        particle = st.selectbox("Select Particle:", particles, key="pseudo_particle")
        
        if st.button("Analyze Pseudorapidity", key="pseudo_btn"):
            with st.spinner("Analyzing pseudorapidity..."):
                try:
                    query = f"""
                        SELECT 
                            SQRT({particle}_PX^2 + {particle}_PY^2 + {particle}_PZ^2) AS p_tot,
                            ATANH({particle}_PZ / SQRT({particle}_PX^2 + {particle}_PY^2 + {particle}_PZ^2)) AS eta,
                            ATAN2({particle}_PY, {particle}_PX) AS phi,
                            SQRT({particle}_PX^2 + {particle}_PY^2) AS pt
                        FROM '{file_path}'
                        WHERE {particle}_PX IS NOT NULL
                            AND {particle}_PY IS NOT NULL
                            AND {particle}_PZ IS NOT NULL
                            AND ABS({particle}_PZ / SQRT({particle}_PX^2 + {particle}_PY^2 + {particle}_PZ^2)) < 1.0
                    """
                    
                    df = con.execute(query).df()
                    df_clean = df.dropna(subset=['eta'])
                    
                    eta = df_clean['eta'].values
                    pt = df_clean['pt'].values / 1000
                    phi = df_clean['phi'].values
                    p_tot = df_clean['p_tot'].values / 1000
                    
                    fig, ax = plt.subplots(2, 2, figsize=(14, 10))
                    
                    # η distribution
                    ax[0, 0].hist(eta, bins=100, histtype='step', color='blue', lw=2.5, edgecolor='blue')
                    ax[0, 0].axvline(np.mean(eta), color='red', linestyle='--', linewidth=2, label=f"Mean: {np.mean(eta):.3f}")
                    ax[0, 0].set_title(f"{particle} Pseudorapidity (η)", fontsize=12, fontweight='bold')
                    ax[0, 0].set_xlabel("Pseudorapidity (η)")
                    ax[0, 0].set_ylabel("Counts")
                    ax[0, 0].grid(True, alpha=0.3)
                    ax[0, 0].legend()
                    ax[0, 0].set_xlim(0, 7)
                    
                    # pT distribution
                    ax[0, 1].hist(pt, bins=100, histtype='step', color='green', lw=2.5, edgecolor='green')
                    ax[0, 1].axvline(np.mean(pt), color='red', linestyle='--', linewidth=2, label=f"Mean: {np.mean(pt):.3f} GeV")
                    ax[0, 1].set_title(f"{particle} Transverse Momentum (pT)", fontsize=12, fontweight='bold')
                    ax[0, 1].set_xlabel("pT [GeV/c]")
                    ax[0, 1].set_ylabel("Counts")
                    ax[0, 1].grid(True, alpha=0.3)
                    ax[0, 1].legend()
                    
                    # φ distribution
                    ax[1, 0].hist(phi, bins=100, histtype='step', color='purple', lw=2.5, edgecolor='purple')
                    ax[1, 0].set_title(f"{particle} Azimuthal Angle (φ)", fontsize=12, fontweight='bold')
                    ax[1, 0].set_xlabel("Azimuthal Angle (radians)")
                    ax[1, 0].set_ylabel("Counts")
                    ax[1, 0].grid(True, alpha=0.3)
                    
                    # η vs pT scatter
                    scatter = ax[1, 1].scatter(eta, pt, alpha=0.3, s=10, c='darkblue')
                    ax[1, 1].set_title(f"{particle} η vs pT Correlation", fontsize=12, fontweight='bold')
                    ax[1, 1].set_xlabel("Pseudorapidity (η)")
                    ax[1, 1].set_ylabel("Transverse Momentum [GeV/c]")
                    ax[1, 1].grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Statistics
                    st.subheader("📊 Statistics")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Pseudorapidity (η)**")
                        st.write(f"- Mean: {np.mean(eta):.4f}")
                        st.write(f"- Std Dev: {np.std(eta):.4f}")
                        st.write(f"- Median: {np.median(eta):.4f}")
                        st.write(f"- Min: {np.min(eta):.4f}")
                        st.write(f"- Max: {np.max(eta):.4f}")
                    
                    with col2:
                        st.write(f"**Transverse Momentum (pT)**")
                        st.write(f"- Mean: {np.mean(pt):.4f} GeV/c")
                        st.write(f"- Std Dev: {np.std(pt):.4f} GeV/c")
                        st.write(f"- Max: {np.max(pt):.4f} GeV/c")
                    
                except Exception as e:
                    st.error(f"❌ Error: {e}")

# ============================================================================
# PAGE: ANGULAR SEPARATION
# ============================================================================
elif page == "🔄 Angular Separation":
    st.header("🔄 Angular Separation Analysis (ΔR)")
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please upload a parquet file in the sidebar first.")
    else:
        con = duckdb.connect()
        particles = get_available_particles(file_path)
        
        st.markdown("""
        Calculate angular separation between two particles:
        **ΔR = √(Δη² + Δφ²)**
        """)
        
        col1, col2 = st.columns(2)
        with col1:
            particle1 = st.selectbox("First Particle:", particles, key="angular_p1")
        with col2:
            particle2 = st.selectbox("Second Particle:", particles, index=1 if len(particles) > 1 else 0, key="angular_p2")
        
        if st.button("Calculate Angular Separation", key="angular_btn"):
            with st.spinner("Calculating angular separation..."):
                try:
                    query1 = f"""
                        SELECT 
                            ATANH({particle1}_PZ / SQRT({particle1}_PX^2 + {particle1}_PY^2 + {particle1}_PZ^2)) AS eta1,
                            ATAN2({particle1}_PY, {particle1}_PX) AS phi1
                        FROM '{file_path}'
                        WHERE {particle1}_PX IS NOT NULL
                            AND {particle1}_PY IS NOT NULL
                            AND {particle1}_PZ IS NOT NULL
                            AND ABS({particle1}_PZ / SQRT({particle1}_PX^2 + {particle1}_PY^2 + {particle1}_PZ^2)) < 1.0
                    """
                    query2 = f"""
                        SELECT
                            ATANH({particle2}_PZ / SQRT({particle2}_PX^2 + {particle2}_PY^2 + {particle2}_PZ^2)) AS eta2,
                            ATAN2({particle2}_PY, {particle2}_PX) AS phi2
                        FROM '{file_path}'
                        WHERE {particle2}_PX IS NOT NULL
                            AND {particle2}_PY IS NOT NULL
                            AND {particle2}_PZ IS NOT NULL
                            AND ABS({particle2}_PZ / SQRT({particle2}_PX^2 + {particle2}_PY^2 + {particle2}_PZ^2)) < 1.0
                    """
                    
                    df_1 = con.execute(query1).df()
                    df_2 = con.execute(query2).df()
                    
                    d_eta = df_1['eta1'].values - df_2['eta2'].values
                    d_phi = df_1['phi1'].values - df_2['phi2'].values
                    d_phi = np.arctan2(np.sin(d_phi), np.cos(d_phi))
                    
                    angular_separation = np.sqrt(d_eta**2 + d_phi**2)
                    
                    counts, bin_edges = np.histogram(angular_separation, bins=100)
                    peak_bin_index = np.argmax(counts)
                    peak_angular = (bin_edges[peak_bin_index] + bin_edges[peak_bin_index + 1]) / 2
                    
                    fig, ax = plt.subplots(figsize=(12, 6))
                    ax.hist(angular_separation, bins=100, histtype='step', color='blue', lw=2.5, edgecolor='blue')
                    ax.axvline(np.mean(angular_separation), color='red', linestyle='--', linewidth=2, 
                              label=f"Mean: {np.mean(angular_separation):.3f}")
                    ax.axvline(peak_angular, color='blue', linestyle='--', linewidth=2, 
                              label=f"Peak: {peak_angular:.3f}")
                    ax.set_title(f"Angular Separation (ΔR) between {particle1} and {particle2}", fontsize=14, fontweight='bold')
                    ax.set_xlabel("Angular Separation (ΔR)", fontsize=12)
                    ax.set_ylabel("Counts", fontsize=12)
                    ax.grid(True, alpha=0.3)
                    ax.legend(fontsize=11)
                    st.pyplot(fig)
                    
                    # Statistics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Mean ΔR", f"{np.mean(angular_separation):.3f}")
                    with col2:
                        st.metric("Peak ΔR", f"{peak_angular:.3f}")
                    with col3:
                        st.metric("Std Dev", f"{np.std(angular_separation):.3f}")
                    with col4:
                        st.metric("Max ΔR", f"{np.max(angular_separation):.3f}")
                    
                except Exception as e:
                    st.error(f"❌ Error: {e}")

# ============================================================================
# PAGE: ML MODELS
# ============================================================================
elif page == "🤖 ML Models":
    st.header("🤖 Machine Learning Models")
    
    st.markdown("""
    This section documents the ML models available in your project for classifying B meson decay events.
    
    Both models are trained on the full dataset and can be used for signal vs. background classification.
    """)
    
    model_tab1, model_tab2 = st.tabs(["XGBoost Regressor", "Boosted Tree + Neural Network"])
    
    with model_tab1:
        st.subheader("🌲 XGBoost Regressor")
        st.markdown("""
        **File:** `xgboostregressor.py`
        
        An extreme gradient boosting model for regression or classification of particle decay events.
        
        **Features:**
        - Fast, scalable gradient boosting
        - Built-in regularization to prevent overfitting
        - Handles high-dimensional feature spaces
        
        **Training Data:**
        - Invariant mass features from all particles
        - Momentum and angular separations
        - Full dataset training
        
        **Usage:**
        ```bash
        python xgboostregressor.py
        ```
        """)
        st.info("💡 XGBoost excels at capturing non-linear relationships in particle physics data.")
    
    with model_tab2:
        st.subheader("🧠 Boosted Tree + Neural Network Hybrid")
        st.markdown("""
        **File:** `boostedtree+neural.py`
        
        A hybrid model combining gradient boosting with deep learning for enhanced classification.
        
        **Architecture:**
        - Gradient boosting for feature extraction
        - Neural network for final classification layer
        - Ensemble methods for robustness
        
        **Training Data:**
        - Multi-scale features (momentum, mass, angles)
        - Signal vs. background discrimination
        - Full dataset training
        
        **Usage:**
        ```bash
        python boostedtree+neural.py
        ```
        """)
        st.info("💡 Hybrid models combine interpretability with deep learning flexibility.")
    
    st.markdown("---")
    st.markdown("""
    ### 📊 Model Selection Guide
    
    | Criterion | XGBoost | Boosted + Neural |
    |-----------|---------|------------------|
    | Speed | ⚡⚡⚡ Very Fast | ⚡⚡ Moderate |
    | Accuracy | ⭐⭐⭐⭐ Excellent | ⭐⭐⭐⭐⭐ Outstanding |
    | Interpretability | ⭐⭐⭐ Good | ⭐⭐ Limited |
    | Data Requirements | 🔽 Lower | 🔼 Higher |
    | Complexity | 📊 Medium | 🧩 Complex |
    """)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📖 Quick Tips")
st.sidebar.markdown("""
- **Bplus:** Parent B meson particle
- **J_psi_1S:** Charmonium intermediate state
- **Kplus:** Kaon hadron
- **mu_plus, mu_minus:** Muon leptons
- **ΔR:** Angular separation in η-φ plane
- **Signal Region:** ~5.28 GeV invariant mass
""")
