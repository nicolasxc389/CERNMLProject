import duckdb
import matplotlib.pyplot as plt
import os

# 1. Initialization and Data Mapping
parquet_file = 'root_data_output.parquet'
if not os.path.exists(parquet_file):
    raise FileNotFoundError(f"Could not find {parquet_file} in the current directory.")

con = duckdb.connect()

# Dynamically pull all prefixes ending with _PX to show the user what's available
prefix_query = f"""
    SELECT REGEXP_EXTRACT(column_name, '^(.*)_PX$', 1) as prefix
    FROM (DESCRIBE SELECT * FROM '{parquet_file}')
    WHERE column_name ILIKE '%_PX'
"""
available_prefixes = [row[0] for row in con.execute(prefix_query).fetchall() if row[0]]

print("="*50)
print("DYNAMICAL CERN DATASET TRACK MAPPER")
print("="*50)
print("Discovered track prefixes in Parquet file:")
for idx, prefix in enumerate(available_prefixes, 1):
    print(f"  [{idx}] {prefix}")
print("-"*50)

# Let SQL do the filtering for columns containing 'P' or 'p'
query_X = """
    SELECT column_name 
    FROM (DESCRIBE SELECT * FROM 'root_data_output.parquet')
    WHERE column_name ILIKE '%_PX%'
"""
query_Y = """
    SELECT column_name 
    FROM (DESCRIBE SELECT * FROM 'root_data_output.parquet')
    WHERE column_name ILIKE '%_PY%'
"""
query_Z = """
    SELECT column_name 
    FROM (DESCRIBE SELECT * FROM 'root_data_output.parquet')
    WHERE column_name ILIKE '%_PZ%'
"""
# Fetch the results straight into a clean Python list
columns_with_momentum_X = con.execute(query_X).fetchall()
columns_with_momentum_Y = con.execute(query_Y).fetchall()
columns_with_momentum_Z = con.execute(query_Z).fetchall()

column_list = [col[0] for col in columns_with_momentum_X + columns_with_momentum_Y + columns_with_momentum_Z]

print("Found momentum-related columns:")
print(column_list)

# 2. Map the particles interactively based on your specific branch names
print("\nPlease map your daughter particles to get the invariant mass of the B meson decay chain (B -> J/psi K).")
mu_plus_prefix  = input("Enter prefix for Muon Plus (e.g., muplus): ").strip()
mu_minus_prefix = input("Enter prefix for Muon Minus (e.g., muminus): ").strip()
kaon_prefix     = input("Enter prefix for Kaon (only Kplus): ").strip()

# PDG Mass Hypotheses (GeV/c^2)
M_MUON = 0.105658
M_KAON = 0.493677

# 3. Constructing the Master Kinetic SQL Query
# We calculate relativistic energies row-by-row inside DuckDB for lightning speed.
master_analysis_query = f"""
    WITH ParticleEnergies AS (
        SELECT 
            -- Geometric Momentum Components and Relativistic Energy for Muon Plus
            {mu_plus_prefix}_PX AS muplus_PX, {mu_plus_prefix}_PY AS muplus_PY, {mu_plus_prefix}_PZ AS muplus_PZ,
            SQRT(POWER({mu_plus_prefix}_PX, 2) + POWER({mu_plus_prefix}_PY, 2) + POWER({mu_plus_prefix}_PZ, 2) + POWER({M_MUON}, 2)) AS muplus_E,
            
            -- Geometric Momentum Components and Relativistic Energy for Muon Minus
            {mu_minus_prefix}_PX AS muminus_PX, {mu_minus_prefix}_PY AS muminus_PY, {mu_minus_prefix}_PZ AS muminus_PZ,
            SQRT(POWER({mu_minus_prefix}_PX, 2) + POWER({mu_minus_prefix}_PY, 2) + POWER({mu_minus_prefix}_PZ, 2) + POWER({M_MUON}, 2)) AS muminus_E,
            
            -- Geometric Momentum Components and Relativistic Energy for the Kaon
            {kaon_prefix}_PX AS Kplus_PX, {kaon_prefix}_PY AS Kplus_PY, {kaon_prefix}_PZ AS Kplus_PZ,
            SQRT(POWER({kaon_prefix}_PX, 2) + POWER({kaon_prefix}_PY, 2) + POWER({kaon_prefix}_PZ, 2) + POWER({M_KAON}, 2)) AS Kplus_E
        FROM '{parquet_file}'
    ),
    CompositeSystems AS (
        SELECT
            -- J/psi system: Add 4-vectors of the two muons
            (muplus_E + muminus_E) AS jpsi_E,
            (muplus_PX + muminus_PX) AS jpsi_PX,
            (muplus_PY + muminus_PY) AS jpsi_PY,
            (muplus_PZ + muminus_PZ) AS jpsi_PZ,
            
            -- B Meson system: Add 4-vectors of both muons AND the kaon
            (muplus_E + muminus_E + Kplus_E) AS B_E,
            (muplus_PX + muminus_PX + Kplus_PX) AS B_PX,
            (muplus_PY + muminus_PY + Kplus_PY) AS B_PY,
            (muplus_PZ + muminus_PZ + Kplus_PZ) AS B_PZ
        FROM ParticleEnergies
    )
    SELECT 
        -- Invariant Mass of J/psi: M = sqrt(E^2 - p^2)
        SQRT(POWER(jpsi_E, 2) - (POWER(jpsi_PX, 2) + POWER(jpsi_PY, 2) + POWER(jpsi_PZ, 2))) AS mass_jpsi,
        
        -- Invariant Mass of B Meson: M = sqrt(E^2 - p^2)
        SQRT(POWER(B_E, 2) - (POWER(B_PX, 2) + POWER(B_PY, 2) + POWER(B_PZ, 2))) AS mass_B
    FROM CompositeSystems
"""
try:
    print("\nProcessing and running relativistic 4-vector algebra over entire dataset...")
    analysis_df = con.execute(master_analysis_query).df()

    # Remove any NaN or unphysical values that could skew plot limits
    analysis_df = analysis_df.dropna()

    # 4. Data Visualization Suite
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: J/psi Mass Spectrum
    ax1.hist(analysis_df['mass_jpsi'], bins=150, histtype='stepfilled', color='royalblue', alpha=0.7, edgecolor='navy', linewidth=1.2)
    ax1.set_title(r'$J/\psi \rightarrow \mu^+\mu^-$ Invariant Mass Peak', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Mass [GeV/c²]', fontsize=11)
    ax1.set_ylabel('Candidates / Bin', fontsize=11)
    ax1.grid(True, alpha=0.3, linestyle='--')
    # Adding a line for expected physical location (~3.097 GeV)
    ax1.axvline(x=3.0969, color='darkred', linestyle=':', label='PDG J/ψ Mass (3.097 GeV)')
    ax1.legend()

    # Plot 2: B Meson Mass Spectrum
    ax2.hist(analysis_df['mass_B'], bins=150, histtype='stepfilled', color='forestgreen', alpha=0.7, edgecolor='darkgreen', linewidth=1.2)
    ax2.set_title(r'$B \rightarrow J/\psi \ K$ Invariant Mass Peak', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Mass [GeV/c²]', fontsize=11)
    ax2.set_ylabel('Candidates / Bin', fontsize=11)
    ax2.grid(True, alpha=0.3, linestyle='--')
    # Adding a line for expected physical location (~5.279 GeV)
    ax2.axvline(x=5.2793, color='darkred', linestyle=':', label='PDG B± Mass (5.279 GeV)')
    ax2.legend()

    
    plt.savefig("invariantmass.png", dpi=300)

except Exception as e:
    print(f"An error occurred during the analysis: {e}")