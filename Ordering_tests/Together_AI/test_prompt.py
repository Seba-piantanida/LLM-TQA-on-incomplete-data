import pandas as pd
import sqlite3
import json
import re
import sqlparse
from sqlparse.sql import IdentifierList, Identifier
from sqlparse.tokens import Keyword, DML


output_schema = """
{
  "table_name": "table name",
  "ordered_entries": [
    {
      entry0
    },
    {
      entry1
    },
    {
      entry2
    }
  ]
}"""


def extract_table_names(sql_query):
    """
    Extract table names from SQL query using sqlparse
    """
    try:
        parsed = sqlparse.parse(sql_query)[0]
        tables = set()
        
        def extract_from_token(token):
            if token.ttype is Keyword and token.value.upper() in ('FROM', 'JOIN', 'INTO', 'UPDATE'):
                return True
            return False
        
        def extract_table_name(token):
            if isinstance(token, Identifier):
                return token.get_name()
            elif token.ttype is None:
                return token.value
            return None
        
        from_seen = False
        for token in parsed.flatten():
            if from_seen:
                if token.ttype is Keyword:
                    from_seen = False
                elif token.ttype is None and token.value.strip():
                    # Clean table name (remove quotes, aliases, etc.)
                    table_name = token.value.strip().split()[0]  # Take first part (before alias)
                    table_name = table_name.strip('`"\'[]')  # Remove quotes
                    if table_name and not table_name.upper() in ('AS', 'ON', 'WHERE', 'GROUP', 'ORDER', 'HAVING', 'LIMIT'):
                        tables.add(table_name)
                        from_seen = False
            elif extract_from_token(token):
                from_seen = True
        
        # Fallback: simple regex extraction if sqlparse doesn't work well
        if not tables:
            # Extract after FROM, JOIN, INTO, UPDATE keywords
            from_pattern = r'(?:FROM|JOIN|INTO|UPDATE)\s+([`"\']?[\w_]+[`"\']?)'
            matches = re.findall(from_pattern, sql_query, re.IGNORECASE)
            for match in matches:
                table_name = match.strip('`"\'[]').split()[0]
                if table_name:
                    tables.add(table_name)
        
        return list(tables)
    
    except Exception as e:
        print(f"Error parsing SQL query: {e}")
        # Fallback to original behavior
        return []


def load_tables_for_query(db_path, sql_query, primary_table, exec_type='normal'):
    """
    Load all tables needed for the SQL query
    """
    # Extract table names from SQL query
    table_names = extract_table_names(sql_query)
    
    # Always include the primary table
    if primary_table not in table_names:
        table_names.append(primary_table)
    
    # If no tables found in parsing, use primary table only
    if not table_names:
        table_names = [primary_table]
    
    tables_data = {}
    
    for table_name in table_names:
        try:
            # Load table from database
            table_data = pd.read_sql_table(table_name, f'sqlite:///{db_path}')
            
            # Apply modifications based on exec_type
            if exec_type == 'remove':
                # Extract ORDER BY column and remove it
                match = re.search(r'ORDER BY\s+([^, \n`]+|`[^`]+`)', sql_query, re.IGNORECASE)
                if match:
                    column = match.group(1).strip('`')
                    table_data = table_data.drop(columns=[column], errors='ignore')
            
            elif exec_type == 'nullable':
                # Extract ORDER BY column and set it to None
                match = re.search(r'ORDER BY\s+([^, \n`]+|`[^`]+`)', sql_query, re.IGNORECASE)
                if match:
                    column = match.group(1).strip('`')
                    if column in table_data.columns:
                        table_data[column] = None
            
            tables_data[table_name] = table_data
            
        except Exception as e:
            print(f"Warning: Could not load table {table_name} from {db_path}: {e}")
            continue
    
    return tables_data


def format_dataset(tables_data, exec_type='normal'):
    """
    Format multiple tables data into a string for the prompt
    """
    formatted_data = ""
    for table_name, table_df in tables_data.items():
        if exec_type == 'normal':
            formatted_data += f"{table_name}: {table_df.to_json(index=False)}\n\n"
        else:
            formatted_data += f"{table_name}: {table_df.to_json()}\n\n"
    
    return formatted_data.strip()


def generate_prompt(question, dataset_str):
    """
    Generate the full prompt that would be sent to the AI model
    """
    return f"""
Objective:
- Respond accurately to the provided query using the given dataset(s).
- If any data fields are NULL, missing, or incomplete, **infer and fill** the missing information with the most logical and contextually appropriate value.
- **Never leave fields empty or set to null.** Always provide the best inferred value based on the dataset context.

Context:
- Here are the dataset(s):
{dataset_str}

Query:
- {question}

Output Format:
- Provide the response strictly in **valid JSON** format.
- Follow exactly this schema:
{output_schema}
- Do not include any explanatory text or notes outside the JSON.
- Ensure that all required fields are completed with non-null values.
"""


def test_prompt_generation(csv_path, exec_type='normal', num_tests=4):
    """
    Test the prompt generation for the first few queries
    """
    # Read the test CSV
    tests = pd.read_csv(csv_path)
    
    print(f"Testing prompt generation for first {num_tests} queries...")
    print(f"Execution type: {exec_type}")
    print("=" * 80)
    
    for i in range(min(num_tests, len(tests))):
        row = tests.iloc[i]
        
        print(f"\n{'='*20} TEST {i+1} {'='*20}")
        print(f"Database: {row['db_path']}")
        print(f"Primary Table: {row['tbl_name']}")
        print(f"SQL Query: {row['query']}")
        print(f"Question: {row['question']}")
        
        # Extract tables from SQL query
        extracted_tables = extract_table_names(row['query'])
        print(f"Extracted Tables: {extracted_tables}")
        
        try:
            # Load all tables for this query
            tables_data = load_tables_for_query(row['db_path'], row['query'], row['tbl_name'], exec_type)
            print(f"Loaded Tables: {list(tables_data.keys())}")
            
            # Show table sizes
            for table_name, table_df in tables_data.items():
                print(f"  - {table_name}: {table_df.shape} (rows, cols)")
            
            # Format dataset
            dataset_str = format_dataset(tables_data, exec_type)
            
            # Generate full prompt
            full_prompt = generate_prompt(row['question'], dataset_str)
            
            print(f"\n--- FULL PROMPT ---")
            print(full_prompt)
            
            # Show first few rows of each table for verification
            print(f"\n--- TABLE DATA PREVIEW ---")
            for table_name, table_df in tables_data.items():
                print(f"\n{table_name} (first 3 rows):")
                print(table_df.head(3).to_string())
            
        except Exception as e:
            print(f"ERROR: {e}")
        
        print("\n" + "="*80)


if __name__ == "__main__":
    # Example usage - modify these parameters as needed
    csv_path = "tests/test_music_cut_10.csv"  # Path to your test CSV file
    exec_type = 'normal'    # Options: 'normal', 'remove', 'nullable'
    
    # Test the first 4 prompts
    test_prompt_generation(csv_path, exec_type, num_tests=4)