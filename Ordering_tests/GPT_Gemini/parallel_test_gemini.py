import pandas as pd
import sqlite3
import re
import json
import os
import time
import google.generativeai as genai
import compute_res
import requests
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import sqlparse
from sqlparse.sql import IdentifierList, Identifier
from sqlparse.tokens import Keyword, DML

# === CONFIG ===
tests_config_path = 'tests_gemini_multi_model.json'  # "tests_gemini_bot_limit_n.json"
GEMINI_MODEL = "gemini-2.5-flash"
wait = 60 # secondi di attesa tra una richiesta e l'altra
attesa_quota_ore = 6  # ore da aspettare solo se quota limit
time_out = 600  # tempo massimo attesa risposta

output_schema = """
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

# Cache per le tabelle caricate (thread-safe usando dizionari separati per thread)
table_caches = {}

# === FUNZIONI UTILI ===
def log(msg):
    print(msg)
    sys.stdout.flush()

def ensure_parent_dir(path: str):
    if path:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)

def extract_json(output: str):
    output = output.strip()
    code_block_match = re.search(r'```(?:json)?\s*([\s\S]+?)\s*```', output)
    if code_block_match:
        json_str = code_block_match.group(1)
    else:
        json_match = re.search(r'(\{[\s\S]+?\})', output)
        json_str = json_match.group(1) if json_match else output
    json_str = re.sub(r'^(json|jsonCopiaModifica)?\s*', '', json_str).strip()
    try:
        parsed_json = json.loads(json_str)
        if "ordered_entries" not in parsed_json:
            raise ValueError("'ordered_entries' missing")
        return parsed_json
    except json.JSONDecodeError as e:
        raise ValueError(f"parsing error:\n{e}\n\nContenuto:\n{json_str[:500]}")

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
        log(f"⚠️ Thread error parsing SQL query: {e}")
        # Fallback to original behavior
        return []

def load_tables_for_query(db_path: str, sql_query: str, primary_table: str, exec_type: str, thread_id: int):
    """
    Load all tables needed for the SQL query (thread-safe version)
    """
    # Get thread-specific cache
    if thread_id not in table_caches:
        table_caches[thread_id] = {}
    table_cache = table_caches[thread_id]
    
    # Extract table names from SQL query
    table_names = extract_table_names(sql_query)
    
    # Always include the primary table
    if primary_table not in table_names:
        table_names.append(primary_table)
    
    # If no tables found in parsing, use primary table only
    if not table_names:
        table_names = [primary_table]
    
    log(f"📊 Thread {thread_id} - Tabelle estratte dalla query: {table_names}")
    
    tables_data = {}
    
    for table_name in table_names:
        cache_key = f"{db_path}:{table_name}:{exec_type}"
        
        # Check cache first
        if cache_key in table_cache:
            log(f"📂 Thread {thread_id} - Caricamento {table_name} dalla cache")
            tables_data[table_name] = table_cache[cache_key]
        else:
            try:
                log(f"📂 Thread {thread_id} - Caricamento {table_name} da {db_path}")
                conn = sqlite3.connect(db_path)
                table_data = pd.read_sql_query(f"SELECT * FROM `{table_name}`", conn)
                conn.close()
                
                # Apply modifications based on exec_type
                match = re.search(r'ORDER BY\s+([^, \n`]+|`[^`]+`)', sql_query, re.IGNORECASE)
                if match:
                    column = match.group(1).strip('`')
                    if exec_type == 'REMOVE':
                        table_data = table_data.drop(columns=[column], errors='ignore')
                        log(f"🗑️ Thread {thread_id} - Rimossa colonna {column} da {table_name}")
                    elif exec_type == 'NULL' and column in table_data.columns:
                        table_data[column] = None
                        log(f"🚫 Thread {thread_id} - Impostata colonna {column} a NULL in {table_name}")
                
                # Cache the table
                table_cache[cache_key] = table_data
                tables_data[table_name] = table_data
                
            except Exception as e:
                log(f"⚠️ Thread {thread_id} - Warning: Could not load table {table_name} from {db_path}: {e}")
                continue
    
    return tables_data

def format_dataset(tables_data: dict, exec_type: str) -> str:
    """
    Format multiple tables data into a string for the prompt
    """
    formatted_data = ""
    for table_name, table_df in tables_data.items():
        if exec_type == 'NORMAL':
            formatted_data += f"{table_name}: {table_df.to_json(index=False)}\n\n"
        else:
            formatted_data += f"{table_name}: {table_df.to_json()}\n\n"
    
    return formatted_data.strip()

def get_dataset(table_name: str, db_path: str, sql_query: str, exec_type: str, thread_id: int = 0) -> str:
    """
    Legacy function - now uses the new multi-table approach
    """
    tables_data = load_tables_for_query(db_path, sql_query, table_name, exec_type, thread_id)
    return format_dataset(tables_data, exec_type)

def load_api_keys(env_path=".env"):
    api_keys = []
    if not Path(env_path).exists():
        raise FileNotFoundError(f"{env_path} non trovato")
    with open(env_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("GEMINI_API_KEY"):
                match = re.search(r"['\"](.*?)['\"]", line)
                if match:
                    api_keys.append(match.group(1))
    if not api_keys:
        raise ValueError("Nessuna API key trovata nel file .env")
    return api_keys

# === FUNZIONE DI ESECUZIONE TEST ===
def run_test(test_cfg, exec_type: str, api_key: str, thread_id: int):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(GEMINI_MODEL)

    test_path = test_cfg["test_path"]
    out_dir = test_cfg["out_dir"]

    log(f"🚀 Thread {thread_id} - Avvio test {test_path} con API key {api_key} modalità {exec_type}")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{os.path.splitext(os.path.basename(test_path))[0]}_{exec_type.lower()}_{GEMINI_MODEL}.csv")
    ensure_parent_dir(out_path)

    # Controlla progresso dal file di output esistente
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        df_out = pd.read_csv(out_path)
        last_completed_idx = df_out.index.max()
    else:
        last_completed_idx = -1

    log(f"Thread {thread_id} - Ultima riga completata nel file di output: {last_completed_idx}")

    df = pd.read_csv(test_path, dtype=str).dropna(subset=['query', 'question'])
    df = df[df.index > last_completed_idx]

    for idx, row in df.iterrows():
        retry_count = 0
        while retry_count < 3:
            try:
                log(f"➡️ Thread {thread_id} - Elaborazione riga {idx} (DB: {row['db_path']}, Table: {row['tbl_name']})")
                
                # Load all tables needed for this query
                tables_data = load_tables_for_query(row['db_path'], row['query'], row['tbl_name'], exec_type, thread_id)
                dataset_str = format_dataset(tables_data, exec_type)
                
                log(f"📋 Thread {thread_id} - Tabelle caricate: {list(tables_data.keys())}")
                for table_name, table_df in tables_data.items():
                    log(f"  - {table_name}: {table_df.shape} (rows, cols)")

                full_prompt = f"""
You are a highly skilled data analyst. Always follow instructions carefully. Always respond strictly in valid JSON.

Objective:
- Respond accurately to the provided query using the given dataset(s).
- If any data fields are NULL, missing, or incomplete, **infer and fill** the missing information with the most logical and contextually appropriate value.
- **Never leave fields empty or set to null.** Always provide the best inferred value based on the dataset context.

Context:
- Here are the dataset(s):
{dataset_str}

Query:
- {row['question']}

Output Format:
- Provide the response strictly in **valid JSON** format.
- Follow exactly this schema:
{output_schema}
- Do not include any explanatory text or notes outside the JSON.
- Ensure that all required fields are completed with non-null values.
"""

                request_options = genai.types.RequestOptions(timeout=time_out)
                response = model.generate_content(full_prompt, request_options=request_options)
                output = response.text
                time.sleep(wait)

                try:
                    parsed = extract_json(output)
                    result_to_save = [[entry[key] for key in entry] for entry in parsed["ordered_entries"]]
                except Exception as e:
                    log(f"❌ Thread {thread_id} - Parsing error at row {idx}: {e}")
                    result_to_save = output

                row_result = {
                    "model": GEMINI_MODEL,
                    "execution_type": exec_type,
                    "db_path": row['db_path'],
                    "table": row['tbl_name'],
                    "sql_query": row['query'],
                    "question": row['question'],
                    "result": result_to_save,
                    "tables_used": list(tables_data.keys())  # New field to track which tables were used
                }

                pd.DataFrame([row_result]).to_csv(out_path, mode='a', index=False, header=not os.path.exists(out_path))

                log(f"✅ Thread {thread_id} - Riga {idx} completata")
                
                break

            except Exception as e:
                err_msg = str(e)
                log(f"⚠️ Thread {thread_id} - Errore: {err_msg}")
                if 'GenerateRequestsPerDayPerProjectPerModel-FreeTier' in err_msg:
                    log(f"⏳ Thread {thread_id} - Limite quota FreeTier raggiunto. Attendo {attesa_quota_ore} ore...")
                    time.sleep(attesa_quota_ore * 3600)
                else:
                    retry_count += 1
                    time.sleep(30)

    if os.path.exists(out_path):
        try:
            compute_res.process_file(out_path)
        except Exception as e:
            log(f"⚠️ Thread {thread_id} - compute_res error: {e}")

    # Log cache statistics
    thread_cache_size = len(table_caches.get(thread_id, {}))
    log(f"📊 Thread {thread_id} - Cache contiene {thread_cache_size} tabelle")

    try:
        requests.post(
            "https://ntfy.zibibbo-house.it.eu.org/Tesi",
            data=f"Thread {thread_id} Test completato su {test_path} in modalità {exec_type}. Cache: {thread_cache_size} tabelle.",
            auth=(f"{os.getenv('NTFY_USR')}", f"{os.getenv('NTFY_PW')}")
        )
        log(f"📬 Thread {thread_id} - Notifica inviata")
    except Exception as e:
        log(f"⚠️ Thread {thread_id} - Errore invio notifica: {e}")

# === WRAPPER PER OGNI API KEY ===
def run_group(api_key, group_tests, thread_id):
    for test_cfg in group_tests:
        modes_to_run = test_cfg.get("modes", ["NORMAL"])
        for mode in modes_to_run:
            run_test(test_cfg, mode, api_key, thread_id)

# === MAIN ===
API_KEYS = load_api_keys()
log(f"Trovate {len(API_KEYS)} API key")

with open(tests_config_path, 'r') as f:
    tests_list = json.load(f)

# divisione gruppi di test per API key
groups = [tests_list[i::len(API_KEYS)] for i in range(len(API_KEYS))]

with ThreadPoolExecutor(max_workers=len(API_KEYS)) as executor:
    futures = []
    for thread_id, (api_key, group) in enumerate(zip(API_KEYS, groups)):
        futures.append(executor.submit(run_group, api_key, group, thread_id))

    for future in as_completed(futures):
        try:
            future.result()
        except Exception as e:
            log(f"❌ Errore in un task parallelo: {e}")

# Print final cache statistics
total_cached_tables = sum(len(cache) for cache in table_caches.values())
log(f"🏁 Esecuzione completata. Cache totale: {total_cached_tables} tabelle across {len(table_caches)} threads")