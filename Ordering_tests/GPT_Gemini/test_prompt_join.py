import pandas as pd
import sqlite3
import re
import json
import os
import sys
import sqlparse
from sqlparse.sql import Identifier
from sqlparse.tokens import Keyword

# === CONFIG ===
tests_config_path = "test_gemini_falsh.json"
OUTPUT_TXT = "output_prompts.txt"

def log(msg):
    """Stampa messaggi e forza il flush per nohup."""
    print(msg)
    sys.stdout.flush()

def extract_table_names(sql_query):
    """
    Estrae i nomi delle tabelle da una query SQL usando sqlparse.
    """
    try:
        parsed = sqlparse.parse(sql_query)[0]
        tables = set()

        def extract_from_token(token):
            return token.ttype is Keyword and token.value.upper() in ('FROM', 'JOIN', 'INTO', 'UPDATE')

        from_seen = False
        for token in parsed.flatten():
            if from_seen:
                if token.ttype is Keyword:
                    from_seen = False
                elif token.ttype is None and token.value.strip():
                    table_name = token.value.strip().split()[0]
                    table_name = table_name.strip('`"\'[]')
                    if table_name and not table_name.upper() in (
                        'AS','ON','WHERE','GROUP','ORDER','HAVING','LIMIT'
                    ):
                        tables.add(table_name)
                        from_seen = False
            elif extract_from_token(token):
                from_seen = True

        if not tables:
            from_pattern = r'(?:FROM|JOIN|INTO|UPDATE)\s+([`"\']?[\w_]+[`"\']?)'
            matches = re.findall(from_pattern, sql_query, re.IGNORECASE)
            for match in matches:
                table_name = match.strip('`"\'[]').split()[0]
                if table_name:
                    tables.add(table_name)

        return list(tables)
    except Exception as e:
        log(f"⚠️ Errore nel parsing SQL: {e}")
        return []

import re
import sqlite3
import pandas as pd
def get_first_order_column(sql_query: str) -> str | None:
    """
    Estrae in modo sicuro la prima colonna dall'ORDER BY.
    Rimuove ASC/DESC, LIMIT, alias, qualificatori t.col, funzioni semplici,
    e tutti gli apici/backtick/brackets spaiati.
    """
    match = re.search(r'ORDER\s+BY\s+(.+?)(?:$|;)', sql_query, re.IGNORECASE)
    if not match:
        return None

    order_clause = match.group(1).strip()

    # Prendi solo la prima colonna (prima virgola)
    first_col = re.split(r',', order_clause)[0].strip()

    # Rimuovi parole chiave SQL
    first_col = re.sub(
        r'\s+(ASC|DESC|NULLS\s+FIRST|NULLS\s+LAST|LIMIT\s+\d+)\b',
        '',
        first_col,
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


def load_tables_for_query(db_path: str, sql_query: str, primary_table: str, exec_type: str):
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

    log(f"📊 Tabelle estratte dalla query: {table_names}")

    # Trova la prima colonna nell'ORDER BY
    first_order_column = get_first_order_column(sql_query)

    tables_data = {}

    for table_name in table_names:
        try:
            log(f"📂 Caricamento {table_name} da {db_path}")
            conn = sqlite3.connect(db_path)
            table_data = pd.read_sql_query(f"SELECT * FROM `{table_name}`", conn)
            conn.close()

            if first_order_column:
                # confronto case-insensitive con i nomi reali delle colonne
                cols_map = {c.lower(): c for c in table_data.columns}
                key = first_order_column.lower()
                if key in cols_map:
                    real_col = cols_map[key]
                    if exec_type.upper() == "REMOVE":
                        table_data = table_data.drop(columns=[real_col], errors='ignore')
                        log(f"🗑️ Rimossa colonna '{real_col}' da {table_name}")
                    elif exec_type.upper() == "NULL":
                        table_data[real_col] = None
                        log(f"🚫 Impostata colonna '{real_col}' a NULL in {table_name}")
                else:
                    log(f"ℹ️ Colonna '{first_order_column}' non trovata in {table_name}")

            # Salva sempre la tabella caricata, anche se la colonna non esiste
            tables_data[table_name] = table_data

        except Exception as e:
            log(f"⚠️ Warning: impossibile caricare la tabella {table_name} da {db_path}: {e}")
            continue

    return tables_data

def format_dataset(tables_data: dict, exec_type: str) -> str:
    """
    Format dei dati tabellari in stringa da inserire nel prompt.
    """
    formatted_data = ""
    for table_name, table_df in tables_data.items():
        if exec_type.upper() == 'NORMAL':
            formatted_data += f"{table_name}: {table_df.to_json(index=False)}\n\n"
        else:
            formatted_data += f"{table_name}: {table_df.to_json()}\n\n"
    return formatted_data.strip()

def run_test(test_path: str, out_dir: str, progress_path: str, exec_type: str, max_prompts: int = 5):
    log(f"🚀 MOCK test {test_path} modalità {exec_type}")

    df = pd.read_csv(test_path, dtype=str).dropna(subset=['query', 'question'])

    prompts_collected = []
    for idx, row in df.iterrows():
        if len(prompts_collected) >= max_prompts:
            break

        db_path = row['db_path']
        table = row['tbl_name']
        sql = row['query']
        prompt = row['question']

        tables_data = load_tables_for_query(db_path, sql, table, exec_type)
        dataset_str = format_dataset(tables_data, exec_type)

        full_prompt = f"""
You are a highly skilled data analyst. Always follow instructions carefully. Always respond strictly in valid JSON.

Objective:
- Respond accurately to the provided query using the given dataset(s).
- If any data fields are NULL, missing, or incomplete, infer and fill the missing information with the most logical value.
- Never leave fields empty or set to null.

Context:
{dataset_str}

Query:
{prompt}

Output Format:
{{
  "table_name": "table name",
  "ordered_entries": [
    {{"entry0"}},
    {{"entry1"}},
    {{"entry2"}}
  ]
}}
"""
        prompts_collected.append(full_prompt.strip())

    with open(OUTPUT_TXT, "a", encoding="utf-8") as f:
        f.write(f"=== Test file: {test_path}, modalità: {exec_type} ===\n\n")
        for i, p in enumerate(prompts_collected, 1):
            f.write(f"--- Prompt {i} ---\n{p}\n\n")

    log(f"✅ Salvati {len(prompts_collected)} prompt da {test_path} in {OUTPUT_TXT}")

# === MAIN ===
with open(tests_config_path, 'r') as f:
    tests_list = json.load(f)

for test_cfg in tests_list:
    modes_to_run = test_cfg.get("modes", ["NORMAL"])
    for mode in modes_to_run:
        run_test(test_cfg["test_path"], test_cfg["out_dir"], test_cfg["progress_path"], mode)