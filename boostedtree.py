#This function finds the training loss between the theoretical and recorded momentum of a chosen particle

import duckdb
import matplotlib
import matplotlib.pyplot as plt

# Connect to the file
con = duckdb.connect()

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
# Display all the available momentum columns to the user
columns_with_momentum_X = con.execute(query_X).fetchall()
columns_with_momentum_Y = con.execute(query_Y).fetchall()
columns_with_momentum_Z = con.execute(query_Z).fetchall()

column_list = [col[0] for col in columns_with_momentum_X + columns_with_momentum_Y + columns_with_momentum_Z]

print("Found momentum-related columns:")
print(column_list)
particle = input('input the particle name (without _PX, _PY, _PZ) for which you want to calculate the training loss: ')

# Master Query to find the overall and transverse momentum of the chosen particle.
query = f"""
    SELECT 
        -- 1. total momentum magnitude
        SQRT({particle}_PX^2 + {particle}_PY^2 + {particle}_PZ^2) AS p_tot,
        -- 2. transverse momentum magnitude
        {particle}_PT as pt,
        -- 3. invariant mass of the B meson decay chain (B -> J/psi K)
        SQRT(POWER(B_E, 2) - (POWER(B_PX, 2) + POWER(B_PY, 2) + POWER(B_PZ, 2))) AS mass_B
    FROM 'root_data_output.parquet'
    WHERE {particle}_PX IS NOT NULL
        AND {particle}_PY IS NOT NULL
        AND {particle}_PZ IS NOT NULL
        AND B_E IS NOT NULL
        AND B_PX IS NOT NULL
        AND B_PY IS NOT NULL
        AND B_PZ IS NOT NULL
"""
# Begin Decision Tree model

