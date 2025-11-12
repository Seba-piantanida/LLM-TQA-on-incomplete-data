import sqlite3
import pandas as pd
from pandas import DataFrame
import shutil
from typing import List, Dict

def process_tables_with_null_columns(db_path: str, new_db_path: str, columns_to_null: List[str]) -> None:
    """
    Process all tables in SQLite database: set specified columns to NULL using pandas and save to new database.
    
    Args:
        db_path: Path to the source SQLite database
        new_db_path: Path for the new SQLite database
        columns_to_null: List of column names to replace with NULL in all tables (if present)
    """
    # Create a copy of the original database (optional)
    shutil.copyfile(db_path, new_db_path)
    
    # Connect to the source database to get table information
    source_conn = sqlite3.connect(db_path)
    
    try:
        # Get list of all tables in the database
        cursor = source_conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [table[0] for table in cursor.fetchall()]
        
        processed_dfs: Dict[str, DataFrame] = {}
        
        for table_name in tables:
            print(f"Processing table: {table_name}")
            df = pd.read_sql_query(f"SELECT * FROM {table_name}", source_conn)
            
            # Find columns to null that exist in this table
            existing_columns = set(df.columns)
            cols_to_null = [col for col in columns_to_null if col in existing_columns]
            
            if not cols_to_null:
                print(f"No columns to null in table {table_name}")
                processed_dfs[table_name] = df
                continue
            
            # Set those columns to NaN (which becomes NULL in SQLite)
            df[cols_to_null] = pd.NA
            print(f"Set columns {cols_to_null} to NULL in table {table_name}")
            
            processed_dfs[table_name] = df
            
    finally:
        source_conn.close()
    
    # Save the processed tables to the new DB
    with sqlite3.connect(new_db_path) as dest_conn:
        for table_name, df in processed_dfs.items():
            df.to_sql(table_name, dest_conn, if_exists='replace', index=False)
            print(f"Saved processed table {table_name} to new database")
    
    print("All tables processed and saved to new database successfully!")

# Configurazione
db_path = "data/db_cut_300.sqlite"
new_db_path = "data/db_cut_300_col_null.sqlite"
columns_to_null = ["Selling Price", "Original Price", "Rating (Out of 5)", "Launched Price (India)", "Model Name", "city",'countryCode','size','age']

# Esecuzione
process_tables_with_null_columns(db_path, new_db_path, columns_to_null)