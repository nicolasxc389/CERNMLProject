"""
CERN B Meson Decay Classification Dashboard - Fixed Version
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import duckdb
import time

st.set_page_config(
    page_title="CERN B Physics Dashboard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main-header {font-size: 2.5rem; font-weight: bold; color: #1f77b4; text-align: center;}
    </style>
""", unsafe_allow_html=True)

# =====================================================================
# SIDEBAR
# =====================================================================
st.sidebar.markdown("## 📁 Data Source")

DEFAULT_URL = "https://huggingface.co/datasets/nicolasxc2089/Bmesondecayparquet/resolve/main/2017_Magnetic_Down_Photon-Photon.root.parquet"

file_source_path = st.sidebar.text_input("Parquet URL / Path", value=DEFAULT_URL)

sample_size = st.sidebar.slider("Sample Size (rows)", 5000, 100000, 30000, step=5000,
                                help="Smaller = faster & safer on Streamlit Cloud")

use_hf_prefix = st.sidebar.checkbox("Use hf:// prefix (recommended for HF)", value=True)

nav_option = st.sidebar.radio(
    "Select View",
    ["🏠 Home", "📊 Feature Analysis", "🤖 Model Performance", 
     "🎯 Event Classification", "📈 Physics Insights", "⚙️ Pipeline Status"]
)

# =====================================================================
# CONNECTION & LOADING
# =====================================================================
@st.cache_resource
def get_db_connection():
    con = duckdb.connect(":memory:")
    con.execute("INSTALL httpfs;")
    con.execute("LOAD httpfs;")
    return con


@st.cache_data(ttl=1800, show_spinner="Loading sampled data from Hugging Face...")
def load_parquet_data(source_path: str, sample_rows: int, use_hf: bool):
    con = get_db_connection()
    path = source_path

    if use_hf and "huggingface.co" in source_path:
        # Convert to native hf:// format
        try:
            # Extract repo and file path
            if "/resolve/main/" in source_path:
                hf_path = source_path.split("/resolve/main/")[-1]
                repo = source_path.split("/datasets/")[1].split("/resolve")[0]
                path = f"hf://{repo}/{hf_path}"
        except:
            pass

    for attempt in range(3):  # Retry up to 3 times
        try:
            query = f"""
            SELECT * FROM read_parquet('{path}')
            LIMIT {sample_rows}
            """
            df = con.execute(query).df()
            
            st.sidebar.success(f"✅ Loaded {len(df):,} rows")
            if len(df) == 0:
                st.sidebar.warning("Loaded 0 rows - file may be empty or inaccessible")
            return df
        except Exception as e:
            error_str = str(e)
            if "TProtocolException" in error_str and attempt < 2:
                st.sidebar.warning(f"Attempt {attempt+1}/3 failed (common HF issue). Retrying...")
                time.sleep(2)
                continue
            else:
                st.error(f"Failed to load data: {error_str[:300]}")
                st.info("Try toggling 'Use hf:// prefix' or reduce sample size further.")
                return None

    return None


def discover_particle_prefixes(df):
    prefixes = set()
    for col in df.columns:
        if any(col.endswith(s) for s in ['_PX', '_PY', '_PZ', '_PE', '_PT', '_M', '_ETA', '_PHI']):
            prefix = col.rsplit('_', 1)[0]
            prefixes.add(prefix)
    return sorted(list(prefixes))


# Physics helper functions (same as before, slightly cleaned)
def calculate_invariant_mass(df, prefix):
    cols = [f'{prefix}_{c}' for c in ['PX','PY','PZ','PE']]
    if all(c in df.columns for c in cols):
        px,py,pz,pe = cols
        return np.sqrt(np.maximum(0, df[pe]**2 - (df[px]**2 + df[py]**2 + df[pz]**2)))
    return None

# ... (add other calculate_ functions similarly - I kept it short here)

# =====================================================================
# PAGES
# =====================================================================

if nav_option == "🏠 Home":
    st.markdown('<div class="main-header">🔬 CERN B Physics Analysis Dashboard</div>', unsafe_allow_html=True)
    st.info("Sampling enabled for the large 2.5GB dataset. Use sidebar controls.")
    # Your original home content here...

elif nav_option == "📊 Feature Analysis":
    st.title("📊 Feature Analysis & Distributions")
    df = load_parquet_data(file_source_path, sample_size, use_hf_prefix)
    if df is None:
        st.stop()
    
    prefixes = discover_particle_prefixes(df)
    st.sidebar.selectbox("Detected Particles", prefixes, key="prefix_sel")
    
    # Your original Feature Analysis code goes here (using df and prefixes)

elif nav_option in ["🤖 Model Performance", "🎯 Event Classification", "📈 Physics Insights"]:
    st.title(nav_option)
    df = load_parquet_data(file_source_path, sample_size, use_hf_prefix)
    if df is None:
        st.warning("Data failed to load. Try smaller sample or toggle hf:// option.")
        st.stop()
    # Add your page-specific code here

elif nav_option == "⚙️ Pipeline Status":
    st.title("⚙️ Pipeline Status & Configuration")
    # Your original pipeline status code (unchanged)

# Footer
st.divider()
st.caption("CERN B Meson Decay Dashboard | Powered by DuckDB + Streamlit")
