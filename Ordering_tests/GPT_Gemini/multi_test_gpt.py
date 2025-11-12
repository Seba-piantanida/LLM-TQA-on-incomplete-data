import pandas as pd
import sqlite3
import re
import json
from playwright.sync_api import sync_playwright
import time
import os
import random
import requests
import gc
import sqlparse
from sqlparse.sql import IdentifierList, Identifier
from sqlparse.tokens import Keyword, DML

# === CONFIG ===
model = 'gpt-5_mini'
progress_path = 'progress_gpt_UDF.json'
tests_json = 'tests_GPT_UDF.json'
out_dir = 'output/UDF_gpt/'
chrome_path = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
chrome_profile = "/Users/seba/Library/Application Support/Google/Chrome/Profile 5"

RESTART_BROWSER_EVERY = 50    # richieste prima di riavvio browser
RESET_CHAT_EVERY = 25         # richieste prima di nuova chat
WAIT_AFTER_REPLY = (20, 25)    # attesa random dopo risposta in secondi
MAX_RETRIES = 3               # maximum number of retry attempts
# =================

# Cache per le tabelle caricate
table_cache = {}

# Carica lista test
with open(tests_json, 'r') as f:
    tests_list = json.load(f)

# Carica progresso
if os.path.exists(progress_path):
    with open(progress_path, 'r') as f:
        progress = json.load(f)
else:
    progress = {}

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
        print(f"⚠️ Error parsing SQL query: {e}")
        # Fallback to original behavior
        return []

def get_first_order_column(sql_query: str) -> str | None:
    """
    Estrae in modo sicuro la prima colonna dall'ORDER BY.
    Gestisce virgole e concatenazioni con '+'. 
    Rimuove ASC/DESC, LIMIT, alias, qualificatori t.col, funzioni semplici,
    e tutti gli apici/backtick/brackets spaiati.
    """
    match = re.search(r'ORDER\s+BY\s+(.+?)(?:$|;)', sql_query, re.IGNORECASE)
    if not match:
        return None

    order_clause = match.group(1).strip()

    # Prendi solo la prima "espressione" separata da virgola
    first_expr = re.split(r',', order_clause)[0].strip()

    # Se c'è un '+', prendi solo la parte prima del '+'
    first_col_part = re.split(r'\+', first_expr)[0].strip()

    # Rimuovi parole chiave SQL
    first_col = re.sub(
        r'\s+(ASC|DESC|NULLS\s+FIRST|NULLS\s+LAST|LIMIT\s+\d+)\b',
        '',
        first_col_part,
        flags=re.IGNORECASE
    )

    # Togli tutti i possibili quote/apici/backtick/brackets all'inizio e alla fine
    first_col = first_col.strip('`"\'[] ')

    # Se è qualificato t.col -> prendi solo col
    if '.' in first_col:
        first_col = first_col.split('.')[-1].strip('`"\'[] ')

    # Se è una funzione semplice FUNC(col) -> prendi l'argomento
    func_match = re.match(r'^[A-Za-z_][\w_]*\(\s*([^)]+)\s*\)$', first_col)
    if func_match:
        first_col = func_match.group(1).strip('`"\'[] ')

    # Se è un numero (ORDER BY 1) -> ignora
    if first_col.isdigit():
        return None

    return first_col or None

def load_tables_for_query(db_path: str, sql_query: str, primary_table: str, exec_type: str, rem_col: list | None = None):
    """
    Carica tutte le tabelle necessarie alla query SQL.
    In base a exec_type:
      - REMOVE: rimuove la prima colonna dell'ORDER BY (se presente in una tabella)
      - NULL: setta a NULL tutti i valori della prima colonna dell'ORDER BY (se presente)
      - NORMAL: non modifica nulla
    """
    # Estrai i nomi delle tabelle dalla query
    table_names = extract_table_names(sql_query)
    if primary_table not in table_names:
        table_names.append(primary_table)
    if not table_names:
        table_names = [primary_table]

    

    # Trova la prima colonna nell'ORDER BY
    if rem_col is None or len(rem_col) == 0:
        first_order_column = get_first_order_column(sql_query)
    else:
        first_order_column = rem_col 

    tables_data = {}

    for table_name in table_names:
        try:
            
            conn = sqlite3.connect(db_path)
            table_data = pd.read_sql_query(f"SELECT * FROM `{table_name}`", conn)
            conn.close()

            if first_order_column:
                # confronto case-insensitive con i nomi reali delle colonne
                cols_map = {c.lower(): c for c in table_data.columns}
                key = first_order_column[0].lower()
                if key in cols_map:
                    real_col = cols_map[key]
                    if exec_type.upper() == "REMOVE":
                        table_data = table_data.drop(columns=[real_col], errors='ignore')
                    
                    elif exec_type.upper() == "NULL":
                        table_data[real_col] = None
                        
              
            tables_data[table_name] = table_data

        except Exception as e:
            
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

# Funzione per estrarre dataset (deprecated, kept for backward compatibility)
def get_dataset(table_name: str, db_path: str, sql_query: str, exec_type: str) -> str:
    print(f"📂 Caricamento dataset {table_name} da {db_path} (metodo legacy)")
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

# Estrai JSON valido dal DOM
def extract_json_from_page(page):
    candidate_selectors = [
        "div[data-message-author-role='assistant'] code",
        "div[data-message-author-role='assistant'] pre",
        "div[data-message-author-role='assistant'] p",
        "div[data-message-author-role='assistant'] div",
        "div[data-message-author-role='assistant'] span"
    ]
    json_candidates = []
    for selector in candidate_selectors:
        elements = page.locator(selector)
        for i in range(elements.count()):
            text = elements.nth(i).evaluate("el => el.innerText || el.textContent")
            if not text or "{" not in text or "}" not in text:
                continue
            clean_text = re.sub(r"^```(?:json)?", "", text.strip(), flags=re.IGNORECASE)
            clean_text = re.sub(r"```$", "", clean_text.strip())
            clean_text = re.sub(r'^(json|jsonCopiaModifica)\s*', '', clean_text).strip()
            try:
                start = clean_text.index("{")
                end = clean_text.rindex("}") + 1
                parsed = json.loads(clean_text[start:end])
                json_candidates.append(parsed)
            except:
                continue
    if json_candidates:
        return json_candidates[-1]
    else:
        raise ValueError("Nessun JSON valido trovato")

# Funzione per aprire browser
def launch_browser(p):
    browser = p.chromium.launch_persistent_context(
        user_data_dir=chrome_profile,
        executable_path=chrome_path,
        headless=False,
        args=["--disable-blink-features=AutomationControlled", "--start-maximized"]
    )
    page = browser.new_page()
    page.goto("https://chat.openai.com")
    page.wait_for_selector('div.ProseMirror[contenteditable="true"]', timeout=60000)
    return browser, page

# Invio system message
def send_system_message(page):
    page.wait_for_selector('div.ProseMirror[contenteditable="true"]', timeout=60000)
    input_box = page.locator('div.ProseMirror[contenteditable="true"]')
    input_box.click()
    page.wait_for_timeout(1000)
    system_msg = """You are a highly skilled data analyst. Always follow instructions carefully. Always respond strictly in valid JSON inside code blocks, never in plain text."""
    input_box.fill(system_msg)
    input_box.press("Enter")
    page.wait_for_timeout(3000)

def process_query_with_retries(page, full_prompt, idx, max_retries=MAX_RETRIES):
    """
    Process a query with retry logic. Returns parsed result or None if all retries fail.
    """
    for attempt in range(max_retries):
        try:
            print(f"🔄 Tentativo {attempt + 1}/{max_retries} per riga {idx}...")
            
            # Conta messaggi esistenti
            prev_count = page.locator("div[data-message-author-role='assistant']").count()

            # Invio prompt
            textarea = page.locator('div.ProseMirror[contenteditable="true"]')
            textarea.wait_for(state="visible", timeout=60000)
            textarea.fill(full_prompt)
            textarea.press("Enter")

            time.sleep(30)
            
            # Attendi nuova risposta
            try:
                page.wait_for_function(
                    f"document.querySelectorAll('div[data-message-author-role=\"assistant\"]').length > {prev_count}",
                    timeout=120000
                )
                page.wait_for_timeout(random.randint(*WAIT_AFTER_REPLY) * 1000)  # attesa tra 20 e 25 sec
            except:
                print(f"⏳ Timeout risposta su tentativo {attempt + 1}")
                if attempt == max_retries - 1:  # Last attempt
                    raise Exception("Timeout finale dopo tutti i tentativi")
                continue
            
            time.sleep(10)
            
            # Estrai e valida JSON
            parsed = extract_json_from_page(page)
            if "ordered_entries" in parsed and isinstance(parsed["ordered_entries"], list):
                result_to_save = [[entry[k] for k in entry] for entry in parsed["ordered_entries"]]
                print(f"✅ Tentativo {attempt + 1} riuscito per riga {idx}")
                return result_to_save
            else:
                raise ValueError("'ordered_entries' missing or invalid")
                
        except Exception as e:
            print(f"❌ Errore tentativo {attempt + 1}/{max_retries} per riga {idx}: {e}")
            if attempt < max_retries - 1:
                print(f"🔄 Riprovo...")
                time.sleep(5)  # Brief pause before retry
            else:
                print(f"💀 Tutti i tentativi falliti per riga {idx}. Salvataggio risultato vuoto.")
                return []
    
    return []

with sync_playwright() as p:
    browser, page = launch_browser(p)
    input("Premi Invio dopo aver fatto login...")
    send_system_message(page)

    count_global = 0

    for test_cfg in tests_list:
        test_file = test_cfg["name"]
        exec_modes = test_cfg.get("modes", ["NORMAL"])

        for EXEC_TYPE in exec_modes:
            test_path = f'tests/test_{test_file}.csv'
            out_path = f'{out_dir}{test_file}/music_null_fix/{test_file}_type_fix_{EXEC_TYPE}_{model}.csv'
            os.makedirs(os.path.dirname(out_path), exist_ok=True)

            last_completed_idx = int(progress.get(f"{test_path}_{EXEC_TYPE}", -1))
            df = pd.read_csv(test_path, dtype=str).dropna(subset=['query', 'question'])
            df = df[df.index > last_completed_idx]

            if df.empty:
                print(f"[SKIP] {test_path} ({EXEC_TYPE}) già completato.")
                continue

            for idx, row in df.iterrows():
                try:
                    # Riavvio browser periodico
                    if count_global > 0 and count_global % RESTART_BROWSER_EVERY == 0:
                        print(">>> Riavvio browser per evitare rallentamenti...")
                        count_global = 0
                        browser.close()
                        browser, page = launch_browser(p)
                        send_system_message(page)

                    # Reset chat periodico
                    elif count_global > 0 and count_global % RESET_CHAT_EVERY == 0:
                        print(">>> Nuova chat per evitare DOM enorme...")
                        time.sleep(40)
                        page.wait_for_selector('a[data-testid="create-new-chat-button"]', timeout=60000)
                        page.click('a[data-testid="create-new-chat-button"]', force=True)
                        page.wait_for_timeout(2000)
                        send_system_message(page)

                    db_path = row['db_path']
                    table = row['tbl_name']
                    sql = row['query']
                    prompt = row['question']
                    
                    # Load all tables needed for this query
                    rem_cols_list = row['rem_col'].split(',') if 'rem_col' in row and row['rem_col'] else None
                    print(rem_cols_list)

                    tables_data = load_tables_for_query(db_path, sql, table, EXEC_TYPE, rem_col=rem_cols_list)
                    dataset_str = format_dataset(tables_data, EXEC_TYPE)
                    
                    print(f"📋 Tabelle caricate: {list(tables_data.keys())}")
                    for table_name, table_df in tables_data.items():
                        print(f"  - {table_name}: {table_df.shape} (rows, cols)")

                    full_prompt = f"""
                                        Objective:
                                        - Respond accurately to the provided query using the given dataset(s).
                                        - If any data fields are NULL, missing, or incomplete, infer and fill the missing information.
                                        - Never leave fields empty or null.
                                        - if a field is missing or not present in the dataset, infer and add it with the most logical value.

                                        Context:
                                        {dataset_str}

                                        Query:
                                        {prompt}

                                        Output Format:
                                        Follow exactly this schema:
                                        "table_name": "table name", "ordered_entries": [ {{ entry0 }}, {{ entry1 }}, {{ entry2 }} ]
                                        """

                    # Process query with retry logic
                    result_to_save = process_query_with_retries(page, full_prompt, idx)

                    # Check if browser needs to be restarted after failed retries
                    if not result_to_save and count_global > 0:
                        print("🔄 Browser restart after failed retries...")
                        browser.close()
                        browser, page = launch_browser(p)
                        send_system_message(page)

                    row_result = {
                        "model": model,
                        "execution_type": EXEC_TYPE,
                        "db_path": db_path,
                        "table": table,
                        "sql_query": sql,
                        "question": prompt,
                        "result": result_to_save,
                        "tables_used": list(tables_data.keys()),  # New field to track which tables were used
                        "retry_count": MAX_RETRIES if not result_to_save else 1  # Track if retries were needed
                    }
                    pd.DataFrame([row_result]).to_csv(out_path, mode='a', index=False, header=not os.path.exists(out_path))
                    progress[f"{test_path}_{EXEC_TYPE}"] = idx
                    with open(progress_path, 'w') as f:
                        json.dump(progress, f)

                    count_global += 1
                    del dataset_str
                    gc.collect()

                except Exception as e:
                    print(f"❌ Errore critico sulla riga {idx}: {e}")
                    # Save empty result for critical errors too
                    row_result = {
                        "model": model,
                        "execution_type": EXEC_TYPE,
                        "db_path": row['db_path'],
                        "table": row['tbl_name'],
                        "sql_query": row['query'],
                        "question": row['question'],
                        "result": [],
                        "tables_used": [],
                        "retry_count": MAX_RETRIES,
                        "error": str(e)
                    }
                    pd.DataFrame([row_result]).to_csv(out_path, mode='a', index=False, header=not os.path.exists(out_path))
                    progress[f"{test_path}_{EXEC_TYPE}"] = idx
                    with open(progress_path, 'w') as f:
                        json.dump(progress, f)
                    continue

    if browser:
        browser.close()

print(f"📊 Cache tabelle finale: {len(table_cache)} entries")

requests.post(
    os.getenv('NTFY_SERVER'),
    data=f"gpt_bot tests completed. Cache: {len(table_cache)} tabelle.",
    auth=(f"{os.getenv('NTFY_USR')}", f"{os.getenv('NTFY_PW')}")
)