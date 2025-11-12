import os
import pandas as pd
import ast
import sqlite3

from qatch.evaluate_dataset.orchestrator_evaluator import OrchestratorEvaluator

# ==========================
# Helper functions
# ==========================

def convert_result_column(result):
    """Converte stringhe di liste in vere liste, se necessario."""
    if isinstance(result, list):
        return result
    try:
        return ast.literal_eval(result)
    except Exception:
        return result

# ==========================
# UDFs
# ==========================

# Date functions (format "dd-mm-yyyy")
def extract_day(date_str):
    if not date_str:
        return None
    try:
        return int(date_str.split('-')[0])
    except Exception:
        return None

def extract_month(date_str):
    if not date_str:
        return None
    try:
        return int(date_str.split('-')[1])
    except Exception:
        return None

def extract_year(date_str):
    if not date_str:
        return None
    try:
        return int(date_str.split('-')[2])
    except Exception:
        return None

# Name functions
def extract_first_name(name):
    if not name:
        return None
    return " ".join(name.strip().split()[:-1]) or name.strip().split()[0]

def extract_last_name(name):
    if not name:
        return None
    parts = name.strip().split()
    return parts[-1] if len(parts) > 1 else None

# Extra
def extract_initials(name):
    if not name:
        return None
    return ''.join([p[0].upper() for p in name.strip().split() if p])

def reverse_string(s):
    if not s:
        return None
    return s[::-1]

def word_count(s):
    if not s:
        return 0
    return len(s.strip().split())

def register_udfs(conn):
    conn.create_function("extract_day", 1, extract_day)
    conn.create_function("extract_month", 1, extract_month)
    conn.create_function("extract_year", 1, extract_year)
    conn.create_function("extract_first_name", 1, extract_first_name)
    conn.create_function("extract_last_name", 1, extract_last_name)
    conn.create_function("extract_initials", 1, extract_initials)
    conn.create_function("reverse_string", 1, reverse_string)
    conn.create_function("word_count", 1, word_count)

# ==========================
# SQL Execution functions
# ==========================

def execute_sql_query(query, db_path):
    """Execute SQL query and return results as list of lists."""
    try:
        conn = sqlite3.connect(db_path)
        register_udfs(conn)
        
        cursor = conn.cursor()
        cursor.execute(query)
        
        # Get results and convert to list of lists (removing dictionary keys)
        results = cursor.fetchall()
        
        conn.close()
        return list(results)  # Convert tuples to lists
        
    except Exception as e:
        print(f"Error executing query: {query[:100]}... Error: {e}")
        return []

# ==========================
# Subclass OrchestratorEvaluator
# ==========================

class OrchestratorEvaluatorWithUDFs(OrchestratorEvaluator):
    """Subclass that registers custom SQLite UDFs on each DB connection."""
    def _connect(self, db_path):
        # Use original connection method if exists, otherwise sqlite3.connect
        conn = getattr(super(), "_connect", lambda path: sqlite3.connect(path))(db_path)
        register_udfs(conn)
        return conn

# ==========================
# Main processing functions
# ==========================

def compute_avg_res(input_csv, output_csv):
    """Calcola medie raggruppate su colonne specifiche."""
    df = pd.read_csv(input_csv)

    group_cols = ['table', 'model']
    avg_cols = [
        'valid_efficiency_score',
        'cell_precision',
        'cell_recall',
        'execution_accuracy',
        'tuple_cardinality',
        'tuple_constraint',
        'tuple_order'
    ]

    summary_df = df.groupby(group_cols)[avg_cols].mean().reset_index()
    summary_df.rename(columns={col: f'avg_{col}' for col in avg_cols}, inplace=True)
    summary_df.to_csv(output_csv, index=False)
    print(f'Average results saved to {output_csv}')

def process_file(input_path):
    """Execute SQL queries, evaluate results, and save to new location without modifying input."""
    print(f'Processing {input_path}...')
    
    # Read the input file
    df = pd.read_csv(input_path)
    
    # Create a copy to avoid modifying the original
    processed_df = df.copy()
    
    # Execute SQL queries and replace results
    for idx, row in processed_df.iterrows():
        if pd.notna(row['sql_query']) and pd.notna(row['db_path']):
            sql_result = execute_sql_query(row['sql_query'], row['db_path'])
            processed_df.at[idx, 'sql_query'] = sql_result
        else:
            processed_df.at[idx, 'sql_query'] = []
    
    # Convert result column if needed
    if 'result' in processed_df.columns:
        processed_df['result'] = processed_df['result'].apply(convert_result_column)

    # Create evaluator and evaluate
    evaluator = OrchestratorEvaluatorWithUDFs()
    res = evaluator.evaluate_df(
        df=processed_df,
        target_col_name='sql_query',  # Now contains executed results
        prediction_col_name='result',
        db_path_name='db_path',
    )

    # Create output directory structure
    # Change 'output' to 'results' in the path
    res_path = input_path.replace('output', 'results')
    os.makedirs(os.path.dirname(res_path), exist_ok=True)
    
    # Save detailed results
    res.to_csv(res_path, index=False)
    print(f'Saved detailed results to {res_path}')

    # Calculate and save averages
    avg_path = res_path.replace('.csv', '_avg.csv')
    compute_avg_res(res_path, avg_path)

    return res_path, avg_path

def process_all_files(input_dir):
    """Scan recursively all CSV files and process them one by one."""
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.csv'):
                input_path = os.path.join(root, file)
                try:
                    process_file(input_path)
                except Exception as e:
                    print(f'Error processing {input_path}: {e}')

# ==========================
# Entry point
# ==========================

if __name__ == '__main__':
    input_dir = 'output/music_UDF_cut_300_null_fix'  # Change this directory as needed
    process_all_files(input_dir)