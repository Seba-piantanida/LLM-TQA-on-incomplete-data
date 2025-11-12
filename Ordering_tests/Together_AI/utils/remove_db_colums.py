import sqlite3
import pandas as pd
from pandas import DataFrame
import shutil
from typing import List, Dict

def process_tables_with_pandas(db_path: str, new_db_path: str, columns_to_delete: List[str]) -> None:
    """
    Process all tables in SQLite database: remove specified columns using pandas and save to new database.
    
    Args:
        db_path: Path to the source SQLite database
        new_db_path: Path for the new SQLite database
        columns_to_delete: List of column names to remove from all tables
    """
    # Create a copy of the original database (optional - we'll overwrite with pandas)
    shutil.copyfile(db_path, new_db_path)
    
    # Connect to the source database to get table information
    source_conn = sqlite3.connect(db_path)
    
    try:
        # Get list of all tables in the database
        cursor = source_conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [table[0] for table in cursor.fetchall()]
        
        # Dictionary to hold all processed DataFrames
        processed_dfs: Dict[str, DataFrame] = {}
        
        # Process each table
        for table_name in tables:
            print(f"Processing table: {table_name}")
            
            # Read the entire table into a DataFrame
            df = pd.read_sql_query(f"SELECT * FROM {table_name}", source_conn)
            
            # Find columns that actually exist in this table
            existing_columns = set(df.columns)
            cols_to_drop = [col for col in columns_to_delete if col in existing_columns]
            
            if not cols_to_drop:
                print(f"No columns to delete in table {table_name}")
                processed_dfs[table_name] = df
                continue
                
            # Drop the specified columns
            df = df.drop(columns=cols_to_drop)
            print(f"Dropped columns {cols_to_drop} from table {table_name}")
            
            # Store the processed DataFrame
            processed_dfs[table_name] = df
            
    finally:
        source_conn.close()
    
    # Save all processed DataFrames to the new database
    with sqlite3.connect(new_db_path) as dest_conn:
        for table_name, df in processed_dfs.items():
            # Save the DataFrame to the new database
            df.to_sql(table_name, dest_conn, if_exists='replace', index=False)
            print(f"Saved processed table {table_name} to new database")
    
    print("All tables processed and saved to new database successfully!")

# Configuration
db_path = "data/db_cut_300.sqlite"
new_db_path = "data/db_cut_300_col_rem.sqlite"
columns_to_delete = ["Selling Price", "Original Price", "Rating (Out of 5)", 'Launched Price (India)', 'Model Name', 'city','countryCode','size','age']

# Execute the function
process_tables_with_pandas(db_path, new_db_path, columns_to_delete)