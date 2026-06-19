"""
CERN B Meson Decay Classification Dashboard
============================================
Interactive Streamlit dashboard for visualizing particle physics pipeline,
ML model performance, and decay event analysis.

Requirements:
    streamlit, pandas, numpy, matplotlib, duckdb, xgboost, scikit-learn, torch
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import duckdb
import os
from pathlib import Path

# Set page config first
st.set_page_config(
    page_title="CERN B Physics Dashboard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
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
# SIDEBAR - FILE UPLOAD & NAVIGATION
# =====================================================================
st.sidebar.markdown("## 📁 Data & Navigation")
uploaded_file = st.sidebar.file_uploader(
    "Upload Parquet file",
    type="parquet",
    help="Select your processed CERN B physics Parquet file"
)

# Navigation tabs
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
    """Create a cached DuckDB connection."""
    return duckdb.connect(memory=True)

@st.cache_data
def load_parquet_data(file_path):
    """Load and cache parquet data."""
    con = get_db_connection()
    try:
        df = con.execute(f"SELECT * FROM read_parquet('{file_path}')").df()
        return df
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

def discover_particle_prefixes(df):
    """Auto-detect particle prefixes from column names."""
    prefixes = set()
    for col in df.columns:
        if col.endswith(('_PX', '_PY', '_PZ', '_PE', '_PT', '_M')):
            prefix = col.rsplit('_', 1)[0]
            prefixes.add(prefix)
    return sorted(list(prefixes))

def calculate_invariant_mass(df, particle_prefix):
    """Calculate invariant mass for a particle."""
    try:
        px_col = f'{particle_prefix}_PX'
        py_col = f'{particle_prefix}_PY'
        pz_col = f'{particle_prefix}_PZ'
        pe_col = f'{particle_prefix}_PE'
        
        if all(col in df.columns for col in [px_col, py_col, pz_col, pe_col]):
            m_inv = np.sqrt(np.maximum(0,
                df[pe_col]**2 - (df[px_col]**2 + df[py_col]**2 + df[pz_col]**2)
            ))
            return m_inv
    except Exception as e:
        st.warning(f"Could not calculate invariant mass: {e}")
    return None

def calculate_pseudorapidity(df, particle_prefix):
    """Calculate pseudorapidity (η) for a particle."""
    try:
        px_col = f'{particle_prefix}_PX'
        py_col = f'{particle_prefix}_PY'
        pz_col = f'{particle_prefix}_PZ'
        
        if all(col in df.columns for col in [px_col, py_col, pz_col]):
            p_tot = np.sqrt(df[px_col]**2 + df[py_col]**2 + df[pz_col]**2)
            eta = np.arctanh(np.clip(df[pz_col] / np.maximum(p_tot, 1e-10), -0.999, 0.999))
            return eta
    except Exception as e:
        st.warning(f"Could not calculate pseudorapidity: {e}")
    return None

def calculate_transverse_momentum(df, particle_prefix):
    """Calculate transverse momentum (pT) for a particle."""
    try:
        px_col = f'{particle_prefix}_PX'
        py_col = f'{particle_prefix}_PY'
        
        if all(col in df.columns for col in [px_col, py_col]):
            pt = np.sqrt(df[px_col]**2 + df[py_col]**2) / 1000  # Convert to GeV
            return pt
    except Exception as e:
        st.warning(f"Could not calculate pT: {e}")
    return None

def calculate_angular_separation(df, particle1_prefix, particle2_prefix):
    """Calculate angular separation (ΔR) between two particles."""
    try:
        eta1_col = f'{particle1_prefix}_ETA'
        phi1_col = f'{particle1_prefix}_PHI'
        eta2_col = f'{particle2_prefix}_ETA'
        phi2_col = f'{particle2_prefix}_PHI'
        
        # If eta/phi columns don't exist, calculate them
        if eta1_col not in df.columns:
            eta1 = calculate_pseudorapidity(df, particle1_prefix)
        else:
            eta1 = df[eta1_col]
        
        if phi1_col not in df.columns:
            phi1 = np.arctan2(df[f'{particle1_prefix}_PY'], df[f'{particle1_prefix}_PX'])
        else:
            phi1 = df[phi1_col]
            
        if eta2_col not in df.columns:
            eta2 = calculate_pseudorapidity(df, particle2_prefix)
        else:
            eta2 = df[eta2_col]
            
        if phi2_col not in df.columns:
            phi2 = np.arctan2(df[f'{particle2_prefix}_PY'], df[f'{particle2_prefix}_PX'])
        else:
            phi2 = df[phi2_col]
        
        if eta1 is not None and eta2 is not None and phi1 is not None and phi2 is not None:
            d_eta = eta1 - eta2
            d_phi = np.arctan2(np.sin(phi1 - phi2), np.cos(phi1 - phi2))
            dr = np.sqrt(d_eta**2 + d_phi**2)
            return dr
    except Exception as e:
        st.warning(f"Could not calculate angular separation: {e}")
    return None

# =====================================================================
# PAGE: HOME
# =====================================================================
if nav_option == "🏠 Home":
    st.markdown('<div class="main-header">🔬 CERN B Physics Analysis Dashboard</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        ### 📚 Project Overview
        **B± → J/ψ(→ μ⁺μ⁻)K± Decay Classification**
        
        Real CERN particle physics data analyzing B meson decays to classify signal vs. background events.
        """)
    
    with col2:
        st.markdown("""
        ### 🎯 Pipeline Stages
        1. Data Loading (Parquet)
        2. Feature Engineering
        3. ML Classification
        4. Performance Analysis
        5. Physics Visualization
        """)
    
    with col3:
        st.markdown("""
        ### 🔍 Key Features
        - Invariant Mass Reconstruction
        - Pseudorapidity (η) Analysis
        - Transverse Momentum (pT)
        - Angular Separation (ΔR)
        - XGBoost + Neural Network
        """)
    
    st.divider()
    
    # Quick Start
    st.subheader("Quick Start")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        1. **Upload a Parquet file** in the sidebar (processed CERN data)
        2. **Select a dashboard view** to explore different analyses
        3. **Interact with filters** to slice and examine the data
        4. **Review metrics** to understand model performance
        
        **Data Requirements:**
        - Particle momentum components (PX, PY, PZ, PE)
        - Track/vertex information (PT, IPCHI2, FDCHI2, etc.)
        - Supports custom particle prefixes (e.g., Bplus_, J_psi_, Kplus_, Muplus_)
        """)
    
    with col2:
        st.info("""
        **Supported Analyses:**
        - Invariant mass peaks
        - Feature distributions
        - Classification scores
        - Model metrics
        """)

# =====================================================================
# PAGE: FEATURE ANALYSIS
# =====================================================================
elif nav_option == "📊 Feature Analysis":
    st.title("📊 Feature Analysis & Distributions")
    
    if uploaded_file is None:
        st.warning("Please upload a Parquet file in the sidebar to begin.")
    else:
        # Save uploaded file temporarily
        temp_path = f"/tmp/{uploaded_file.name}"
        with open(temp_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        
        # Load data
        df = load_parquet_data(temp_path)
        
        if df is not None:
            prefixes = discover_particle_prefixes(df)
            
            st.sidebar.markdown("### Feature Analysis Options")
            analysis_type = st.sidebar.radio(
                "Choose Analysis",
                ["Invariant Mass", "Pseudorapidity", "Transverse Momentum", "Angular Separation"]
            )
            
            # ===== INVARIANT MASS ANALYSIS =====
            if analysis_type == "Invariant Mass":
                st.subheader("Invariant Mass Reconstruction")
                
                particle = st.selectbox("Select Particle", prefixes)
                
                m_inv = calculate_invariant_mass(df, particle)
                
                if m_inv is not None:
                    # Remove NaN values
                    m_inv_clean = m_inv[~np.isnan(m_inv)]
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Mean Mass", f"{np.mean(m_inv_clean):.2f} MeV/c²")
                    with col2:
                        st.metric("Peak Mass", f"{m_inv_clean[np.argmax(np.histogram(m_inv_clean, bins=100)[0])]:.2f} MeV/c²")
                    with col3:
                        st.metric("Std Dev", f"{np.std(m_inv_clean):.2f} MeV/c²")
                    with col4:
                        st.metric("N Events", f"{len(m_inv_clean):,}")
                    
                    # Plot
                    fig, ax = plt.subplots(figsize=(12, 5))
                    counts, bins, _ = ax.hist(m_inv_clean, bins=100, alpha=0.7, color='darkorange', edgecolor='black')
                    ax.axvline(np.mean(m_inv_clean), color='red', linestyle='--', linewidth=2, label=f"Mean: {np.mean(m_inv_clean):.2f} MeV/c²")
                    ax.axvline(np.median(m_inv_clean), color='blue', linestyle='--', linewidth=2, label=f"Median: {np.median(m_inv_clean):.2f} MeV/c²")
                    ax.set_xlabel("Invariant Mass [MeV/c²]")
                    ax.set_ylabel("Counts")
                    ax.set_title(f"{particle} Invariant Mass Distribution")
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                    
                    # Statistics
                    with st.expander("📋 Detailed Statistics"):
                        st.write(f"""
                        - **Mean:** {np.mean(m_inv_clean):.4f} MeV/c²
                        - **Median:** {np.median(m_inv_clean):.4f} MeV/c²
                        - **Std Dev:** {np.std(m_inv_clean):.4f} MeV/c²
                        - **Min:** {np.min(m_inv_clean):.4f} MeV/c²
                        - **Max:** {np.max(m_inv_clean):.4f} MeV/c²
                        - **25th Percentile:** {np.percentile(m_inv_clean, 25):.4f} MeV/c²
                        - **75th Percentile:** {np.percentile(m_inv_clean, 75):.4f} MeV/c²
                        """)
            
            # ===== PSEUDORAPIDITY ANALYSIS =====
            elif analysis_type == "Pseudorapidity":
                st.subheader("Pseudorapidity (η) Analysis")
                
                particle = st.selectbox("Select Particle", prefixes)
                
                eta = calculate_pseudorapidity(df, particle)
                pt = calculate_transverse_momentum(df, particle)
                
                if eta is not None and pt is not None:
                    eta_clean = eta[~np.isnan(eta)]
                    pt_clean = pt[~np.isnan(pt)]
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Mean η", f"{np.mean(eta_clean):.3f}")
                    with col2:
                        st.metric("Mean pT", f"{np.mean(pt_clean):.2f} GeV/c")
                    with col3:
                        st.metric("Max pT", f"{np.max(pt_clean):.2f} GeV/c")
                    with col4:
                        st.metric("N Events", f"{len(eta_clean):,}")
                    
                    # 2x2 Subplots
                    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
                    
                    # Pseudorapidity
                    axes[0, 0].hist(eta_clean, bins=100, color='blue', alpha=0.7, edgecolor='black')
                    axes[0, 0].axvline(np.mean(eta_clean), color='red', linestyle='--', linewidth=2, label=f"Mean: {np.mean(eta_clean):.3f}")
                    axes[0, 0].set_title(f"{particle} Pseudorapidity (η)")
                    axes[0, 0].set_xlabel("η")
                    axes[0, 0].set_ylabel("Counts")
                    axes[0, 0].legend()
                    axes[0, 0].grid(True, alpha=0.3)
                    
                    # Transverse Momentum
                    axes[0, 1].hist(pt_clean, bins=100, color='green', alpha=0.7, edgecolor='black')
                    axes[0, 1].axvline(np.mean(pt_clean), color='red', linestyle='--', linewidth=2, label=f"Mean: {np.mean(pt_clean):.2f} GeV/c")
                    axes[0, 1].set_title(f"{particle} Transverse Momentum (pT)")
                    axes[0, 1].set_xlabel("pT [GeV/c]")
                    axes[0, 1].set_ylabel("Counts")
                    axes[0, 1].legend()
                    axes[0, 1].grid(True, alpha=0.3)
                    
                    # η vs pT scatter
                    axes[1, 0].scatter(eta_clean, pt_clean, alpha=0.3, s=5, color='darkblue')
                    axes[1, 0].set_title(f"{particle} η vs pT Correlation")
                    axes[1, 0].set_xlabel("η")
                    axes[1, 0].set_ylabel("pT [GeV/c]")
                    axes[1, 0].grid(True, alpha=0.3)
                    
                    # 2D Histogram
                    h = axes[1, 1].hist2d(eta_clean, pt_clean, bins=50, cmap='YlOrRd')
                    axes[1, 1].set_title(f"{particle} η-pT Density")
                    axes[1, 1].set_xlabel("η")
                    axes[1, 1].set_ylabel("pT [GeV/c]")
                    plt.colorbar(h[3], ax=axes[1, 1])
                    
                    plt.tight_layout()
                    st.pyplot(fig)
            
            # ===== TRANSVERSE MOMENTUM ANALYSIS =====
            elif analysis_type == "Transverse Momentum":
                st.subheader("Transverse Momentum (pT) Analysis")
                
                particle = st.selectbox("Select Particle", prefixes)
                
                pt = calculate_transverse_momentum(df, particle)
                
                if pt is not None:
                    pt_clean = pt[~np.isnan(pt)]
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Mean pT", f"{np.mean(pt_clean):.2f} GeV/c")
                    with col2:
                        st.metric("Median pT", f"{np.median(pt_clean):.2f} GeV/c")
                    with col3:
                        st.metric("Max pT", f"{np.max(pt_clean):.2f} GeV/c")
                    with col4:
                        st.metric("N Events", f"{len(pt_clean):,}")
                    
                    fig, ax = plt.subplots(figsize=(12, 5))
                    ax.hist(pt_clean, bins=100, alpha=0.7, color='green', edgecolor='black')
                    ax.axvline(np.mean(pt_clean), color='red', linestyle='--', linewidth=2, label=f"Mean: {np.mean(pt_clean):.2f} GeV/c")
                    ax.set_xlabel("Transverse Momentum [GeV/c]")
                    ax.set_ylabel("Counts")
                    ax.set_title(f"{particle} Transverse Momentum Distribution")
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
            
            # ===== ANGULAR SEPARATION ANALYSIS =====
            elif analysis_type == "Angular Separation":
                st.subheader("Angular Separation (ΔR) Analysis")
                
                col1, col2 = st.columns(2)
                with col1:
                    particle1 = st.selectbox("First Particle", prefixes)
                with col2:
                    particle2 = st.selectbox("Second Particle", prefixes)
                
                if particle1 != particle2:
                    dr = calculate_angular_separation(df, particle1, particle2)
                    
                    if dr is not None:
                        dr_clean = dr[~np.isnan(dr)]
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Mean ΔR", f"{np.mean(dr_clean):.3f}")
                        with col2:
                            st.metric("Median ΔR", f"{np.median(dr_clean):.3f}")
                        with col3:
                            st.metric("Max ΔR", f"{np.max(dr_clean):.3f}")
                        with col4:
                            st.metric("N Events", f"{len(dr_clean):,}")
                        
                        fig, ax = plt.subplots(figsize=(12, 5))
                        ax.hist(dr_clean, bins=100, alpha=0.7, color='purple', edgecolor='black')
                        ax.axvline(np.mean(dr_clean), color='red', linestyle='--', linewidth=2, label=f"Mean: {np.mean(dr_clean):.3f}")
                        ax.set_xlabel("Angular Separation (ΔR)")
                        ax.set_ylabel("Counts")
                        ax.set_title(f"Angular Separation between {particle1} and {particle2}")
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                        st.pyplot(fig)
                else:
                    st.error("Please select two different particles.")

# =====================================================================
# PAGE: MODEL PERFORMANCE
# =====================================================================
elif nav_option == "🤖 Model Performance":
    st.title("🤖 ML Model Performance & Diagnostics")
    
    if uploaded_file is None:
        st.warning("Please upload a Parquet file in the sidebar.")
    else:
        temp_path = f"/tmp/{uploaded_file.name}"
        with open(temp_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        
        df = load_parquet_data(temp_path)
        
        if df is not None:
            st.subheader("Model Information")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.info("**XGBoost Classifier**\n- Boosted Decision Tree\n- 150 estimators\n- Max depth: 4")
            with col2:
                st.info("**Deep Neural Network**\n- PyTorch\n- 64→32→1 layers\n- 40 epochs")
            with col3:
                st.info("**Training Data**\n- Signal: B mass peak\n- Background: Sidebands\n- 70/30 train/test split")
            
            st.divider()
            
            # Feature selection for analysis
            prefixes = discover_particle_prefixes(df)
            
            st.subheader("Feature Importance & Distribution")
            
            with st.expander("ℹ️ How to Interpret"):
                st.markdown("""
                **Feature Importance** shows which variables the XGBoost model relies on most for classification.
                
                **Expected High-Importance Features:**
                - B meson kinematics (pT, momentum)
                - J/ψ vertex quality (χ²)
                - Kaon identification (PIDK)
                - Muon isolation scores (IPCHI2)
                """)
            
            # Display available columns
            st.markdown("### Available Columns in Dataset")
            col_display = st.columns(4)
            cols = df.columns.tolist()
            for idx, col in enumerate(cols):
                with col_display[idx % 4]:
                    st.code(col, language="text")

# =====================================================================
# PAGE: EVENT CLASSIFICATION
# =====================================================================
elif nav_option == "🎯 Event Classification":
    st.title("🎯 Signal vs. Background Classification")
    
    if uploaded_file is None:
        st.warning("Please upload a Parquet file in the sidebar.")
    else:
        temp_path = f"/tmp/{uploaded_file.name}"
        with open(temp_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        
        df = load_parquet_data(temp_path)
        
        if df is not None:
            st.subheader("Classification Strategy")
            
            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown("""
                ### Peak Identification
                **Signal Region:**
                - ±2σ from peak
                - Pure signal events
                
                **Background Region:**
                - ±5σ from peak
                - Sidebands only
                """)
            
            with col2:
                prefixes = discover_particle_prefixes(df)
                if len(prefixes) > 0:
                    particle = st.selectbox("Analyze particle for mass peak:", prefixes)
                    
                    m_inv = calculate_invariant_mass(df, particle)
                    
                    if m_inv is not None:
                        m_inv_clean = m_inv[~np.isnan(m_inv)]
                        
                        # Find peak
                        counts, bin_edges = np.histogram(m_inv_clean, bins=100)
                        peak_idx = np.argmax(counts)
                        peak_mass = (bin_edges[peak_idx] + bin_edges[peak_idx+1]) / 2
                        sigma = np.std(m_inv_clean) / 4.0
                        
                        st.markdown(f"""
                        ### Peak Statistics
                        - **Peak Mass:** {peak_mass:.2f} MeV/c²
                        - **Sigma:** {sigma:.2f} MeV/c²
                        - **Signal Range:** [{peak_mass - 2*sigma:.2f}, {peak_mass + 2*sigma:.2f}] MeV/c²
                        - **Background Range:** [{peak_mass - 5*sigma:.2f}, {peak_mass + 5*sigma:.2f}] MeV/c²
                        """)
                        
                        # Visualization
                        fig, ax = plt.subplots(figsize=(12, 5))
                        ax.hist(m_inv_clean, bins=100, alpha=0.6, color='gray', label='All Events')
                        
                        sig_mask = (m_inv_clean > peak_mass - 2*sigma) & (m_inv_clean < peak_mass + 2*sigma)
                        ax.hist(m_inv_clean[sig_mask], bins=50, alpha=0.8, color='green', label='Signal Region (±2σ)')
                        
                        ax.axvline(peak_mass, color='red', linestyle='--', linewidth=2, label='Peak')
                        ax.axvline(peak_mass - 2*sigma, color='blue', linestyle=':', linewidth=1.5, alpha=0.7)
                        ax.axvline(peak_mass + 2*sigma, color='blue', linestyle=':', linewidth=1.5, alpha=0.7)
                        ax.axvline(peak_mass - 5*sigma, color='orange', linestyle=':', linewidth=1.5, alpha=0.7)
                        ax.axvline(peak_mass + 5*sigma, color='orange', linestyle=':', linewidth=1.5, alpha=0.7)
                        
                        ax.set_xlabel("Invariant Mass [MeV/c²]")
                        ax.set_ylabel("Counts")
                        ax.set_title(f"{particle} Mass Peak with Signal/Background Regions")
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                        st.pyplot(fig)
                        
                        # Classification metrics
                        n_signal = np.sum(sig_mask)
                        n_bkg = len(m_inv_clean) - n_signal
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Signal Events", f"{n_signal:,}")
                        with col2:
                            st.metric("Background Events", f"{n_bkg:,}")
                        with col3:
                            st.metric("Signal Fraction", f"{100*n_signal/len(m_inv_clean):.1f}%")

# =====================================================================
# PAGE: PHYSICS INSIGHTS
# =====================================================================
elif nav_option == "📈 Physics Insights":
    st.title("📈 Physics Analysis & Insights")
    
    if uploaded_file is None:
        st.warning("Please upload a Parquet file in the sidebar.")
    else:
        temp_path = f"/tmp/{uploaded_file.name}"
        with open(temp_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        
        df = load_parquet_data(temp_path)
        
        if df is not None:
            st.subheader("Decay Channel: B± → J/ψ(→ μ⁺μ⁻)K±")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("""
                ### Physics Context
                **B Meson (B±)**
                - Contains bottom quark (b)
                - Essential for CP-violation studies
                - Rare decay → precision probe
                
                **J/ψ Meson**
                - Charmonium state (cc̄)
                - Decays to μ⁺μ⁻ (clean signature)
                - Sharp mass peak at 3.1 GeV/c²
                """)
            
            with col2:
                st.markdown("""
                ### Key Observables
                **Invariant Mass**
                - B meson peak ≈ 5.28 GeV/c²
                - Mass reconstruction from 4-vectors
                - Signal/background separation
                
                **Kinematics**
                - pT: perpendicular to beam
                - η: pseudorapidity (detector coords)
                - Angular correlations (ΔR)
                """)
            
            st.divider()
            
            # Data-driven insights
            st.subheader("Data-Driven Analysis")
            
            prefixes = discover_particle_prefixes(df)
            
            if len(prefixes) > 0:
                # Create multi-column correlation
                st.markdown("### Multi-Particle Kinematics")
                
                selected_particles = st.multiselect(
                    "Select particles to correlate:",
                    prefixes,
                    default=prefixes[:2] if len(prefixes) >= 2 else prefixes
                )
                
                if len(selected_particles) >= 2:
                    # Calculate multiple features for selected particles
                    features_dict = {}
                    
                    for particle in selected_particles:
                        eta = calculate_pseudorapidity(df, particle)
                        pt = calculate_transverse_momentum(df, particle)
                        m_inv = calculate_invariant_mass(df, particle)
                        
                        if eta is not None:
                            features_dict[f'{particle}_eta'] = eta[~np.isnan(eta)]
                        if pt is not None:
                            features_dict[f'{particle}_pt'] = pt[~np.isnan(pt)]
                        if m_inv is not None:
                            features_dict[f'{particle}_m_inv'] = m_inv[~np.isnan(m_inv)]
                    
                    if len(features_dict) > 0:
                        # Display statistics
                        st.markdown("#### Kinematic Summary")
                        summary_data = []
                        for key, values in features_dict.items():
                            summary_data.append({
                                "Feature": key,
                                "Mean": np.mean(values),
                                "Std": np.std(values),
                                "Min": np.min(values),
                                "Max": np.max(values),
                                "Median": np.median(values)
                            })
                        
                        summary_df = pd.DataFrame(summary_data)
                        st.dataframe(summary_df, use_container_width=True)

# =====================================================================
# PAGE: PIPELINE STATUS
# =====================================================================
elif nav_option == "⚙️ Pipeline Status":
    st.title("⚙️ Pipeline Status & Configuration")
    
    st.subheader("Processing Pipeline")
    
    pipeline_stages = [
        ("parquet.py", "Load ROOT → Parquet", "✅ Complete"),
        ("momentum.py", "Momentum Distributions", "✅ Complete"),
        ("pseudorapidity.py", "η, φ, pT Features", "✅ Complete"),
        ("invariantmass.py", "4-Vector Mass Reconstruction", "✅ Complete"),
        ("boostedtree+neural.py", "XGBoost + DNN Classification", "✅ Complete"),
        ("angularseparation.py", "Angular Separation Analysis", "✅ Complete"),
        ("xgboostregressor.py", "Resolution Sharpening", "🔨 In Progress"),
        ("dashboard.py", "Interactive Visualization", "✅ Complete"),
    ]
    
    for script, description, status in pipeline_stages:
        col1, col2, col3 = st.columns([2, 3, 1])
        with col1:
            st.code(script, language="text")
        with col2:
            st.write(description)
        with col3:
            st.write(status)
    
    st.divider()
    
    st.subheader("Data Flow")
    st.markdown("""
    ```
    ROOT Files (CERN OpenData)
         ↓
    parquet.py → Parquet Files
         ↓
    Feature Engineering (momentum, pseudorapidity, invariantmass, angular_separation)
         ↓
    boostedtree+neural.py → Signal/Background Classification
         ↓
    xgboostregressor.py → Mass Resolution Improvement
         ↓
    dashboard.py ← Interactive Visualization & Analysis
    ```
    """)
    
    st.divider()
    
    st.subheader("Required Dependencies")
    
    dependencies = {
        "Data Processing": ["pandas>=1.3.0", "numpy>=1.21.0", "duckdb>=0.5.0", "uproot>=4.0.0"],
        "Visualization": ["matplotlib>=3.4.0", "streamlit>=1.0.0"],
        "Machine Learning": ["scikit-learn>=1.0.0", "xgboost>=1.5.0", "torch>=1.9.0"],
    }
    
    for category, packages in dependencies.items():
        with st.expander(f"📦 {category}"):
            for package in packages:
                st.write(f"• `{package}`")
    
    st.divider()
    
    st.subheader("Running the Dashboard")
    st.code(
        "streamlit run dashboard.py",
        language="bash"
    )
    
    st.info("""
    **Next Steps:**
    1. Prepare your CERN Parquet file with processed data
    2. Run: `streamlit run dashboard.py`
    3. Upload your data file using the sidebar file uploader
    4. Explore features, models, and physics insights
    """)

# =====================================================================
# FOOTER
# =====================================================================
st.divider()
st.markdown("""
<div style='text-align: center; color: #888; margin-top: 2rem;'>
    <p><strong>CERN B Physics Dashboard</strong> | High Energy Particle Collision ML Project</p>
    <p>Data Source: <a href='https://opendata.cern.ch/'>CERN OpenData Portal</a></p>
</div>
""", unsafe_allow_html=True)
