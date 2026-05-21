import duckdb

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
# Fetch the results straight into a clean Python list
columns_with_momentum_X = con.execute(query_X).fetchall()
columns_with_momentum_Y = con.execute(query_Y).fetchall()
columns_with_momentum_Z = con.execute(query_Z).fetchall()

column_list = [col[0] for col in columns_with_momentum_X + columns_with_momentum_Y + columns_with_momentum_Z]

print("Found momentum-related columns:")
print(column_list)