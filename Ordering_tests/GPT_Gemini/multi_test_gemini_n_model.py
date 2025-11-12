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
from dotenv import load_dotenv
import sqlparse
from sqlparse.sql import IdentifierList, Identifier
from sqlparse.tokens import Keyword, DML

# === CONFIG ===
tests_config_path = "tests_gemini_multi_model.json"
DEFAULT_GEMINI_MODEL = "gemini-2.5-pro"  # Fallback model
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
print(API_KEY)
attesa = 6  # ore
time_out = 1200  #tempo massimo attesa risposta

# === INIT GOOGLE GEMINI ===


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

# Cache per le tabelle caricate
table_cache = {}

def log(msg):
    """Stampa messaggi e forza il flush per nohup."""
    print(msg)
    sys.stdout.flush()

def ensure_parent_dir(path: str):
    """Crea la cartella padre se non esiste."""
    if path is None or path.strip() == "":
        return
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
        log(f"⚠️ Error parsing SQL query: {e}")
        # Fallback to original behavior
        return []

def load_tables_for_query(db_path: str, sql_query: str, primary_table: str, exec_type: str):
    """
    Carica tutte le tabelle necessarie alla query SQL.
    In base a exec_type:
      - REMOVE: rimuove la prima colonna dell'ORDER BY
      - NULL: setta a NULL tutti i valori della prima colonna dell'ORDER BY
      - NORMAL: non modifica nulla
    """
    # Estrai i nomi delle tabelle dalla query
    table_names = extract_table_names(sql_query)

    # Assicurati di includere sempre la tabella primaria
    if primary_table not in table_names:
        table_names.append(primary_table)

    # Se non viene trovata nessuna tabella, usa solo la primaria
    if not table_names:
        table_names = [primary_table]

    log(f"📊 Tabelle estratte dalla query: {table_names}")

    # Trova la parte dell'ORDER BY
    order_match = re.search(r'ORDER BY\s+([^\n;]+)', sql_query, re.IGNORECASE)
    first_order_column = None
    if order_match:
        order_clause = order_match.group(1)
        # Prendi la prima colonna, splittando per virgole o +
        first_order_column = re.split(r'[,+]', order_clause)[0].strip()
        # Rimuovi eventuali ASC/DESC e backtick
        first_order_column = re.sub(r'\s+(ASC|DESC)$', '', first_order_column, flags=re.IGNORECASE).strip('` ')

    tables_data = {}

    for table_name in table_names:
        try:
            log(f"📂 Caricamento {table_name} da {db_path}")
            conn = sqlite3.connect(db_path)
            table_data = pd.read_sql_query(f"SELECT * FROM `{table_name}`", conn)
            conn.close()

            # Se c'è una colonna ORDER BY, applica modifiche in base a exec_type
            if first_order_column and first_order_column in table_data.columns:
                if exec_type.upper() == "REMOVE":
                    table_data = table_data.drop(columns=[first_order_column], errors='ignore')
                    log(f"🗑️ Rimossa colonna {first_order_column} da {table_name}")
                elif exec_type.upper() == "NULL":
                    table_data[first_order_column] = None
                    log(f"🚫 Impostata colonna {first_order_column} a NULL in {table_name}")
                elif exec_type.upper() == "NORMAL":
                    log(f"✅ Nessuna modifica applicata a {first_order_column} in {table_name}")

            tables_data[table_name] = table_data

        except Exception as e:
            log(f"⚠️ Warning: impossibile caricare la tabella {table_name} da {db_path}: {e}")
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

def get_dataset(table_name: str, db_path: str, sql_query: str, exec_type: str) -> str:
    """
    Deprecated: Use load_tables_for_query and format_dataset instead.
    Kept for backward compatibility.
    """
    log(f"📂 Caricamento dataset {table_name} da {db_path} (metodo legacy)")
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(f"SELECT * FROM `{table_name}`", conn)
    conn.close()
    match = re.search(r'ORDER BY\s+([^, \n`]+|`[^`]+`)', sql_query, re.IGNORECASE)
    if match:
        column = match.group(1).strip('`')
        if exec_type == 'REMOVE':
            df = df.drop(columns=[column], errors='ignore')
        elif exec_type == 'NULL' and column in df.columns:
            df[column] = None
    return f"{table_name}: {df.to_json(index=False)}\n"

def run_test(test_path: str, out_dir: str, progress_path: str, exec_type: str, model_name: str):
    log(f"🚀 Avvio test {test_path} modalità {exec_type} con modello {model_name}")

    # Initialize the model for this specific run
    try:
        model = genai.GenerativeModel(model_name)
        log(f"🤖 Modello {model_name} inizializzato correttamente")
    except Exception as e:
        log(f"❌ Errore nell'inizializzazione del modello {model_name}: {e}")
        return

    # crea cartella di output
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(
        out_dir,
        f"{os.path.splitext(os.path.basename(test_path))[0]}_{exec_type.lower()}_{model_name}.csv"
    )

    ensure_parent_dir(out_path)
    ensure_parent_dir(progress_path)

    # gestione progress - include model name in progress key
    progress_key = f"{test_path}_{exec_type}_{model_name}"
    
    if os.path.exists(progress_path) and os.path.getsize(progress_path) > 0:
        with open(progress_path, 'r') as f:
            try:
                progress = json.load(f)
            except json.JSONDecodeError:
                log(f"⚠️ File {progress_path} corrotto o vuoto. Inizializzo da zero.")
                progress = {}
    else:
        progress = {}

    last_completed_idx = int(progress.get(progress_key, -1))
    log(f"Ultima riga completata per {model_name}: {last_completed_idx}")

    df = pd.read_csv(test_path, dtype=str).dropna(subset=['query', 'question'])
    df = df[df.index > last_completed_idx]

    for idx, row in df.iterrows():
        retry_count = 0
        while retry_count < 3:
            try:
                log(f"➡️ Elaborazione riga {idx} con {model_name} (DB: {row['db_path']}, Table: {row['tbl_name']})")

                db_path = row['db_path']
                table = row['tbl_name']
                sql = row['query']
                prompt = row['question']
                
                # Load all tables needed for this query
                tables_data = load_tables_for_query(db_path, sql, table, exec_type)
                dataset_str = format_dataset(tables_data, exec_type)
                
                log(f"📋 Tabelle caricate: {list(tables_data.keys())}")
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
- {prompt}

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

                try:
                    parsed = extract_json(output)
                    entries = parsed["ordered_entries"]
                    array_of_arrays = [[entry[key] for key in entry] for entry in entries]
                    result_to_save = array_of_arrays
                except Exception as e:
                    log(f"❌ Parsing error at row {idx}: {e}")
                    result_to_save = output

                row_result = {
                    "model": model_name,
                    "execution_type": exec_type,
                    "db_path": db_path,
                    "table": table,
                    "sql_query": sql,
                    "question": prompt,
                    "result": result_to_save,
                    "tables_used": list(tables_data.keys())  # New field to track which tables were used
                }

                pd.DataFrame([row_result]).to_csv(
                    out_path, mode='a', index=False, header=not os.path.exists(out_path)
                )

                progress[progress_key] = idx
                with open(progress_path, 'w') as f:
                    json.dump(progress, f)

                log(f"✅ Riga {idx} completata con {model_name} e salvata in {out_path}")
                time.sleep(15)
                break  # riga completata, esci dal retry loop

            except Exception as e:
                error_msg = str(e)
                log(f"⚠️ Errore con {model_name}: {error_msg}")

                if "429" in error_msg and "quota" in error_msg.lower() and 'perday' in error_msg.lower():
                    log(f"⏳ Limite quota raggiunto per {model_name}. Attendo {attesa} ore...")
                    time.sleep(attesa * 60 * 60)
                else:
                    retry_count += 1
                    log(f"⏳ Tentativo {retry_count}/3 per riga {idx} con {model_name}")
                    if retry_count >= 3:
                        log(f"❌ Saltata riga {idx} dopo 3 tentativi con {model_name}")
                        row_result = {
                        "model": model_name,
                        "execution_type": exec_type,
                        "db_path": db_path,
                        "table": table,
                        "sql_query": sql,
                        "question": prompt,
                        "result": [],
                        "tables_used": list(tables_data.keys())  # New field to track which tables were used
                        }

                        pd.DataFrame([row_result]).to_csv(
                            out_path, mode='a', index=False, header=not os.path.exists(out_path)
                        )

                        progress[progress_key] = idx
                        with open(progress_path, 'w') as f:
                            json.dump(progress, f)
                            break
                    time.sleep(30)

    log(f"🎯 Output salvato in {out_path}")
    
    # --- FIX: process_file solo se il file esiste ---
    if os.path.exists(out_path):
        try:
            compute_res.process_file(out_path)
        except Exception as e:
            log(f"⚠️ Errore in compute_res.process_file: {e}")
    else:
        ensure_parent_dir(out_path)
        log(f"⚠️ File {out_path} non trovato, salto compute_res.process_file")

    # notifica
    try:
        requests.post(
            "https://ntfy.zibibbo-house.it.eu.org/Tesi",
            data=f"Test completato su {test_path} in modalità {exec_type} con modello {model_name}. Cache: {len(table_cache)} tabelle.",
            auth=(f"{os.getenv('NTFY_USR')}", f"{os.getenv('NTFY_PW')}")
        )
        log(f"📬 Notifica inviata")
    except Exception as e:
        log(f"⚠️ Errore invio notifica: {e}")

# === MAIN ===
with open(tests_config_path, 'r') as f:
    config = json.load(f)

# Extract models list from config, with fallback to default
if isinstance(config, dict) and "models" in config:
    models_list = config.get("models", [DEFAULT_GEMINI_MODEL])
    tests_list = config.get("tests", [])
elif isinstance(config, list):
    # Legacy format - assume it's just a list of tests
    log(f"⚠️ Legacy config format detected. Using default model: {DEFAULT_GEMINI_MODEL}")
    models_list = [DEFAULT_GEMINI_MODEL]
    tests_list = config
else:
    log(f"❌ Invalid config format. Using defaults.")
    models_list = [DEFAULT_GEMINI_MODEL]
    tests_list = []

log(f"🤖 Modelli configurati: {models_list}")
log(f"📋 Test configurati: {len(tests_list)}")

# Run each test with each model
for test_cfg in tests_list:
    modes_to_run = test_cfg.get("modes", ["NORMAL"])
    for mode in modes_to_run:
        for model_name in models_list:
            log(f"🔄 Esecuzione: {test_cfg['test_path']} | Modalità: {mode} | Modello: {model_name}")
            run_test(test_cfg["test_path"], test_cfg["out_dir"], test_cfg["progress_path"], mode, model_name)