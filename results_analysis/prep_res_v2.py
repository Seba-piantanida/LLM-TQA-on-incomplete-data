#!/usr/bin/env python3
"""
Script per unire file CSV con due formati diversi in un formato standardizzato.
Prende tutti i CSV in una directory (escludendo quelli con 'avg' nel nome)
e li converte nel formato target standardizzato, poi li separa per categoria.
Ora include estrazione automatica delle tabelle dalle query SQL e dal linguaggio naturale.
Con logica migliorata per la categorizzazione delle query ORDER BY.
"""

import pandas as pd
import os
import glob
import logging
from pathlib import Path
from typing import List, Dict, Optional, Set
import argparse
import re
import sqlite3
from pathlib import Path

from pathlib import Path

def get_all_csv_files(base_dir="."):
    """
    Trova ricorsivamente tutti i file CSV in una directory base,
    ignorando i file che iniziano con '._'.
    
    Args:
        base_dir: La cartella principale da cui iniziare la ricerca.
        
    Returns:
        Una lista di percorsi completi ai file CSV validi.
    """
    lista_file_csv = []
    logger.debug(f"Inizio ricerca CSV in '{base_dir}'...")
    for cartella, sottocartelle, files in os.walk(base_dir):
        for nome_file in files:
            # Cerca i file che finiscono con '.csv' E che NON iniziano con '._'
            if nome_file.endswith('.csv') and not nome_file.startswith('._'):
                # Costruisce il percorso completo del file e lo aggiunge alla lista
                percorso_completo = os.path.join(cartella, nome_file)
                lista_file_csv.append(percorso_completo)
    return lista_file_csv
# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Definizione dei formati
FORMAT_1_COLUMNS = [
    "model", "execution_type", "db_path", "table", "sql_query", "question", 
    "result", "valid_efficiency_score", "cell_precision", "cell_recall", 
    "execution_accuracy", "tuple_cardinality", "tuple_constraint", "tuple_order"
]

FORMAT_2_COLUMNS = [
    "db_path", "table_name", "test_category", "query", "model", "AI_answer", 
    "SQL_query", "failed_attempts", "valid_efficiency_score", "cell_precision", 
    "cell_recall", "execution_accuracy", "tuple_cardinality", "tuple_constraint", 
    "tuple_order"
]

# Formato target (basato su FORMAT_2 con aggiunta execution_type)
TARGET_COLUMNS = [
    "db_path", "table_name", "test_category", "query", "model", "AI_answer", 
    "SQL_query", "failed_attempts", "execution_type", "valid_efficiency_score", 
    "cell_precision", "cell_recall", "execution_accuracy", "tuple_cardinality", 
    "tuple_constraint", "tuple_order"
]

# Categorie di separazione
CATEGORIES = ["normal", "null", "remove", "clean_null", "clean_remove"]

# Cache per le tabelle dei database
DB_TABLES_CACHE = {}

def get_database_tables(db_path: str) -> Set[str]:
    """
    Ottiene la lista delle tabelle da un database SQLite.
    Usa una cache per evitare di riaprire lo stesso database più volte.
    
    Args:
        db_path: Percorso al database SQLite
        
    Returns:
        Set di nomi delle tabelle nel database
    """
    if not db_path or pd.isna(db_path):
        return set()
    
    # Controlla la cache
    if db_path in DB_TABLES_CACHE:
        return DB_TABLES_CACHE[db_path]
    
    tables = set()
    
    try:
        # Converti il percorso relativo in assoluto se necessario
        if not os.path.isabs(db_path):
            # Prova diversi percorsi base comuni
            possible_paths = [
                db_path,
                os.path.join(".", db_path),
                os.path.join("..", db_path),
                os.path.join("data", os.path.basename(db_path))
            ]
            
            db_exists = False
            for path in possible_paths:
                if os.path.exists(path):
                    db_path = path
                    db_exists = True
                    break
            
            if not db_exists:
                logger.warning(f"Database non trovato: {db_path}")
                DB_TABLES_CACHE[db_path] = tables
                return tables
        
        # Connessione al database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Query per ottenere i nomi delle tabelle
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0].lower() for row in cursor.fetchall()}
        
        conn.close()
        logger.debug(f"Tabelle trovate in {db_path}: {tables}")
        
    except Exception as e:
        logger.warning(f"Errore nell'accesso al database {db_path}: {e}")
    
    # Salva nella cache
    DB_TABLES_CACHE[db_path] = tables
    return tables

def extract_tables_from_sql(sql_query: str, db_tables: Set[str] = None) -> List[str]:
    """
    Estrae i nomi delle tabelle da una query SQL.
    
    Args:
        sql_query: La query SQL
        db_tables: Set delle tabelle disponibili nel database (opzionale)
        
    Returns:
        Lista dei nomi delle tabelle trovate
    """
    if not sql_query or pd.isna(sql_query):
        return []
    
    tables = []
    sql_lower = str(sql_query).lower()
    
    # Pattern per estrarre tabelle dopo FROM, JOIN, UPDATE, INSERT INTO, DELETE FROM
    patterns = [
        r'from\s+([a-zA-Z_][a-zA-Z0-9_]*)',
        r'join\s+([a-zA-Z_][a-zA-Z0-9_]*)',
        r'update\s+([a-zA-Z_][a-zA-Z0-9_]*)',
        r'insert\s+into\s+([a-zA-Z_][a-zA-Z0-9_]*)',
        r'delete\s+from\s+([a-zA-Z_][a-zA-Z0-9_]*)'
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, sql_lower)
        for match in matches:
            table_name = match.strip()
            if table_name and table_name not in ['select', 'where', 'group', 'order', 'having']:
                # Se abbiamo la lista delle tabelle del DB, verifica che esista
                if db_tables is None or table_name in db_tables:
                    if table_name not in tables:
                        tables.append(table_name)
    
    return tables

def extract_tables_from_natural_language(query: str, db_tables: Set[str] = None) -> List[str]:
    """
    Estrae i nomi delle tabelle da una query in linguaggio naturale.
    Cerca stringhe tra apici singoli e verifica se esistono come tabelle nel database.
    
    Args:
        query: La query in linguaggio naturale
        db_tables: Set delle tabelle disponibili nel database
        
    Returns:
        Lista dei nomi delle tabelle trovate
    """
    if not query or pd.isna(query):
        return []
    
    tables = []
    
    # Pattern per trovare stringhe tra apici singoli
    pattern = r"'([^']*)'"
    matches = re.findall(pattern, str(query))
    
    for match in matches:
        table_candidate = match.strip().lower()
        
        # Se abbiamo la lista delle tabelle del DB, verifica che esista
        if db_tables and table_candidate in db_tables:
            if table_candidate not in tables:
                tables.append(table_candidate)
        elif db_tables is None and table_candidate:
            # Se non abbiamo la lista delle tabelle, accetta candidati ragionevoli
            if len(table_candidate) > 2 and table_candidate.isalnum():
                if table_candidate not in tables:
                    tables.append(table_candidate)
    
    # Pattern aggiuntivi per table references comuni
    additional_patterns = [
        r'`([^`]+)`',  # Backticks
        r'\btable\s+([a-zA-Z_][a-zA-Z0-9_]*)',  # "table tablename"
        r'from\s+the\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+table',  # "from the tablename table"
    ]
    
    for pattern in additional_patterns:
        matches = re.findall(pattern, str(query), re.IGNORECASE)
        for match in matches:
            table_candidate = match.strip().lower()
            if db_tables and table_candidate in db_tables:
                if table_candidate not in tables:
                    tables.append(table_candidate)
    
    return tables

def fill_missing_table_info(df: pd.DataFrame) -> pd.DataFrame:
    """
    Riempie le informazioni mancanti nella colonna table_name estraendo
    le tabelle dalle query SQL o dalle query in linguaggio naturale.
    
    Args:
        df: DataFrame con i dati
        
    Returns:
        DataFrame con le informazioni delle tabelle completate
    """
    logger.info("Riempiendo informazioni mancanti delle tabelle...")
    
    # Conta i record che necessitano di elaborazione
    mask_missing = df['table_name'].isna() | (df['table_name'] == '') | (df['table_name'] == 'nan')
    missing_count = mask_missing.sum()
    
    if missing_count == 0:
        logger.info("Nessuna tabella mancante trovata")
        return df
    
    logger.info(f"Trovati {missing_count} record con informazioni tabella mancanti")
    
    filled_count = 0
    sql_filled = 0
    nl_filled = 0
    
    for idx, row in df.iterrows():
        if not mask_missing.iloc[idx]:
            continue
        
        # Ottieni le tabelle del database se possibile
        db_tables = set()
        if 'db_path' in df.columns and pd.notna(row.get('db_path')):
            db_tables = get_database_tables(str(row['db_path']))
        
        extracted_tables = []
        
        # Prova prima con la query SQL
        if 'SQL_query' in df.columns and pd.notna(row.get('SQL_query')):
            sql_tables = extract_tables_from_sql(str(row['SQL_query']), db_tables)
            if sql_tables:
                extracted_tables.extend(sql_tables)
                sql_filled += 1
                logger.debug(f"Tabelle estratte da SQL per riga {idx}: {sql_tables}")
        
        # Se non trovato nelle SQL, prova con la query in linguaggio naturale
        if not extracted_tables and 'query' in df.columns and pd.notna(row.get('query')):
            nl_tables = extract_tables_from_natural_language(str(row['query']), db_tables)
            if nl_tables:
                extracted_tables.extend(nl_tables)
                nl_filled += 1
                logger.debug(f"Tabelle estratte da linguaggio naturale per riga {idx}: {nl_tables}")
        
        # Aggiorna il campo table_name se trovate delle tabelle
        if extracted_tables:
            # Rimuovi duplicati mantenendo l'ordine
            unique_tables = []
            seen = set()
            for table in extracted_tables:
                if table not in seen:
                    unique_tables.append(table)
                    seen.add(table)
            
            # Formatta come richiesto: "tab1, tab2, tab3"
            table_string = ", ".join(unique_tables)
            df.at[idx, 'table_name'] = table_string
            filled_count += 1
            
            logger.debug(f"Riga {idx}: tabelle impostate a '{table_string}'")
    
    logger.info(f"Completato riempimento tabelle:")
    logger.info(f"  - Record processati: {missing_count}")
    logger.info(f"  - Record riempiti: {filled_count}")
    logger.info(f"    - Da query SQL: {sql_filled}")
    logger.info(f"    - Da linguaggio naturale: {nl_filled}")
    logger.info(f"  - Record ancora vuoti: {missing_count - filled_count}")
    
    return df

def detect_csv_format(file_path: str) -> int:
    """
    Rileva il formato del CSV analizzando le colonne.
    
    Returns:
        1: Formato 1 (con execution_type)
        2: Formato 2 (con test_category)
        0: Formato sconosciuto
    """
    try:
        # Leggi solo la prima riga per ottenere le colonne
        df_sample = pd.read_csv(file_path, nrows=0)
        columns = set(df_sample.columns.str.strip().str.lower())
        
        # Controlla caratteristiche distintive
        format_1_indicators = {"execution_type", "question", "result"}
        format_2_indicators = {"test_category", "ai_answer", "failed_attempts"}
        
        format_1_score = len([col for col in format_1_indicators if col in columns])
        format_2_score = len([col for col in format_2_indicators if col in columns])
        
        if format_1_score >= 2:
            return 1
        elif format_2_score >= 2:
            return 2
        else:
            logger.warning(f"Formato non riconosciuto per {file_path}")
            return 0
            
    except Exception as e:
        logger.error(f"Errore nel rilevamento formato per {file_path}: {e}")
        return 0

def infer_execution_type_from_filename(filename: str) -> str:
    """
    Inferisce l'execution_type dal nome del file.
    """
    filename_lower = filename.lower()
    
    for category in CATEGORIES:
        if category in filename_lower:
            return category
    
    # Fallback: prova a inferire da pattern comuni
    if "clean" in filename_lower and "null" in filename_lower:
        return "clean_null"
    elif "clean" in filename_lower and ("remove" in filename_lower or "rem" in filename_lower):
        return "clean_remove"
    elif "null" in filename_lower:
        return "null"
    elif "remove" in filename_lower or 'rem':
        return "remove"
    else:
        return "normal"

def categorize_sql_query(sql_query: str) -> str:
    """
    Categorizza una query SQL basandosi sui pattern specifici richiesti.
    Logica migliorata per le query ORDER BY, incluse operazioni aritmetiche.
    
    Args:
        sql_query: La query SQL da analizzare
        
    Returns:
        Categoria della query
    """
    if pd.isna(sql_query) or not sql_query:
        return "UNKNOWN"
    
    # Pulisci e normalizza la query
    query_clean = re.sub(r'\s+', ' ', str(sql_query).strip())
    query_lower = query_clean.lower()
    
    # Rimuovi commenti SQL
    query_lower = re.sub(r'--.*?$', '', query_lower, flags=re.MULTILINE)
    query_lower = re.sub(r'/\*.*?\*/', '', query_lower, flags=re.DOTALL)
    query_lower = query_lower.strip()
    
    # Pattern per identificare SELECT con ORDER BY e LIMIT
    select_orderby_limit_pattern = r'select\s+(.+?)\s+from\s+.+?\s+order\s+by\s+(.+?)(?:\s+limit\s+\d+)?(?:\s*;?\s*$)'
    match = re.search(select_orderby_limit_pattern, query_lower, re.DOTALL)
    
    if match and ("order by" in query_lower and "limit" in query_lower):
        select_clause = match.group(1).strip()
        order_by_clause = match.group(2).strip()
        
        logger.debug(f"Analizzando query ORDER BY - SELECT: '{select_clause}', ORDER BY: '{order_by_clause}'")
        
        # Controlla se è SELECT *
        is_select_all = select_clause.strip() == '*'
        
        # Analizza l'ORDER BY per operazioni aritmetiche e multiple colonne
        column_count = analyze_order_by_complexity(order_by_clause)
        
        logger.debug(f"SELECT all: {is_select_all}, ORDER BY column count: {column_count}")
        
        if is_select_all:
            # SELECT * FROM ... ORDER BY
            if column_count >= 2:  # 2 o più colonne (incluse operazioni aritmetiche)
                return "ORDERBY-ADVANCED"
            elif column_count == 1:
                return "ORDERBY-SINGLE"
            else:
                return "ORDERBY"
        else:
            # SELECT col1, col2, ... FROM ... ORDER BY (projection)
            return "ORDERBY-PROJECT"
    
    # Fallback per altre query con ORDER BY ma senza LIMIT
    elif "order by" in query_lower:
        # Stessa logica ma senza requisito del LIMIT
        select_orderby_pattern = r'select\s+(.+?)\s+from\s+.+?\s+order\s+by\s+(.+?)(?:\s*;?\s*$)'
        match = re.search(select_orderby_pattern, query_lower, re.DOTALL)
        
        if match:
            select_clause = match.group(1).strip()
            order_by_clause = match.group(2).strip()
            
            is_select_all = select_clause.strip() == '*'
            column_count = analyze_order_by_complexity(order_by_clause)
            
            if is_select_all:
                if column_count >= 2:  # 2 o più colonne (incluse operazioni aritmetiche)
                    return "ORDERBY-ADVANCED"
                elif column_count == 1:
                    return "ORDERBY-SINGLE"
                else:
                    return "ORDERBY"
            else:
                return "ORDERBY-PROJECT"
        else:
            return "ORDERBY"
    
    # Altre categorie esistenti
    elif "group by" in query_lower:
        return "GROUPBY"
    elif "join" in query_lower:
        return "JOIN"
    elif "where" in query_lower:
        return "FILTER"
    elif "sum" in query_lower or "avg" in query_lower or "count" in query_lower or "max" in query_lower or "min" in query_lower:
        return "AGGREGATE"
    elif "distinct" in query_lower:
        return "DISTINCT"
    elif "union" in query_lower:
        return "UNION"
    elif "having" in query_lower:
        return "HAVING"
    elif "limit" in query_lower:
        return "LIMIT"
    else:
        return "BASIC"


def analyze_order_by_complexity(order_by_clause: str) -> int:
    """
    Analizza la complessità di una clausola ORDER BY per determinare
    il numero effettivo di "elementi" di ordinamento.
    
    Gestisce:
    - Colonne separate da virgole: col1, col2, col3
    - Operazioni aritmetiche: col1 + col2 + col3
    - Espressioni complesse con parentesi
    - Combinazioni di operatori (+, -, *, /)
    
    Args:
        order_by_clause: La clausola ORDER BY da analizzare
        
    Returns:
        Numero di elementi di ordinamento identificati
    """
    if not order_by_clause:
        return 0
    
    # Rimuovi ASC/DESC dalla fine per semplificare l'analisi
    clause_clean = re.sub(r'\s+(asc|desc)\s*$', '', order_by_clause.strip(), flags=re.IGNORECASE)
    
    # Conta le virgole che separano espressioni di ordinamento diverse
    # (ignora quelle dentro parentesi che potrebbero essere parametri di funzioni)
    comma_separated_expressions = []
    paren_count = 0
    current_expr = ""
    
    for char in clause_clean:
        if char == '(':
            paren_count += 1
        elif char == ')':
            paren_count -= 1
        elif char == ',' and paren_count == 0:
            if current_expr.strip():
                comma_separated_expressions.append(current_expr.strip())
            current_expr = ""
            continue
        
        current_expr += char
    
    # Aggiungi l'ultima espressione
    if current_expr.strip():
        comma_separated_expressions.append(current_expr.strip())
    
    # Se ci sono più espressioni separate da virgole, conta quelle
    if len(comma_separated_expressions) > 1:
        return len(comma_separated_expressions)
    
    # Altrimenti, analizza l'unica espressione per operazioni aritmetiche
    single_expression = comma_separated_expressions[0] if comma_separated_expressions else clause_clean
    
    # Conta i riferimenti a colonne in operazioni aritmetiche
    # Pattern per identificare colonne/campi (nomi tra backticks, apici, o identificatori validi)
    column_patterns = [
        r'`([^`]+)`',  # Colonne tra backticks: `Nome`
        r"'([^']+)'",  # Colonne tra apici singoli: 'Nome'
        r'"([^"]+)"',  # Colonne tra doppi apici: "Nome"
        r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b'  # Identificatori semplici: Nome, Classe
    ]
    
    # Estrai tutti i possibili riferimenti a colonne
    potential_columns = set()
    
    for pattern in column_patterns:
        matches = re.findall(pattern, single_expression)
        for match in matches:
            # Filtra parole chiave SQL comuni
            if match.lower() not in ['and', 'or', 'not', 'null', 'true', 'false', 'asc', 'desc']:
                potential_columns.add(match)
    
    # Se abbiamo operatori aritmetici, conta le colonne coinvolte
    arithmetic_operators = ['+', '-', '*', '/', '%']
    has_arithmetic = any(op in single_expression for op in arithmetic_operators)
    
    if has_arithmetic and len(potential_columns) >= 2:
        # Per operazioni aritmetiche come "Nome + Classe + Durata", 
        # conta il numero di colonne coinvolte nell'operazione
        return len(potential_columns)
    
    # Se non ci sono operazioni aritmetiche o una sola colonna, è un ordinamento semplice
    return 1 if potential_columns else 0
def infer_category_from_natural_language(query_text: str) -> str:
    """
    Inferisce la categoria da una query in linguaggio naturale.
    
    Args:
        query_text: Testo della query in linguaggio naturale
        
    Returns:
        Categoria inferita
    """
    if pd.isna(query_text):
        return "UNKNOWN"
    
    query_lower = str(query_text).lower()
    
    # Pattern per ORDER BY
    if ("order" in query_lower and "by" in query_lower) or ("sort" in query_lower):
        if "limit" in query_lower or "first" in query_lower or "top" in query_lower:
            # Prova a determinare se è single, advanced o project basandosi su indizi nel testo
            if "all columns" in query_lower or "all fields" in query_lower or "everything" in query_lower:
                if "multiple" in query_lower or "several" in query_lower or " and " in query_lower:
                    return "ORDERBY-ADVANCED"
                else:
                    return "ORDERBY-SINGLE"
            else:
                return "ORDERBY-PROJECT"
        else:
            return "ORDERBY"
    elif "group" in query_lower and "by" in query_lower:
        return "GROUPBY"
    elif "join" in query_lower or "combine" in query_lower or "merge" in query_lower:
        return "JOIN"
    elif "where" in query_lower or "filter" in query_lower or "condition" in query_lower:
        return "FILTER"
    elif any(word in query_lower for word in ["sum", "average", "count", "total", "maximum", "minimum", "avg", "max", "min"]):
        return "AGGREGATE"
    elif "distinct" in query_lower or "unique" in query_lower:
        return "DISTINCT"
    elif "union" in query_lower:
        return "UNION"
    elif "having" in query_lower:
        return "HAVING"
    elif "limit" in query_lower or "first" in query_lower or "top" in query_lower:
        return "LIMIT"
    else:
        return "BASIC"

def convert_format_1_to_target(df: pd.DataFrame, filename: str = "") -> pd.DataFrame:
    """
    Converte un DataFrame dal formato 1 al formato target.
    """
    logger.info("Convertendo dal formato 1 al formato target...")
    
    # Crea DataFrame target con colonne vuote
    result_df = pd.DataFrame(columns=TARGET_COLUMNS)
    
    # Mappatura diretta delle colonne
    column_mapping = {
        "db_path": "db_path",
        "table": "table_name",
        "sql_query": "SQL_query", 
        "question": "query",
        "model": "model",
        "execution_type": "execution_type",
        "valid_efficiency_score": "valid_efficiency_score",
        "cell_precision": "cell_precision",
        "cell_recall": "cell_recall",
        "execution_accuracy": "execution_accuracy",
        "tuple_cardinality": "tuple_cardinality",
        "tuple_constraint": "tuple_constraint",
        "tuple_order": "tuple_order"
    }
    
    # Copia le colonne mappate
    for source_col, target_col in column_mapping.items():
        if source_col in df.columns:
            result_df[target_col] = df[source_col]
    
    # Gestisci colonne specifiche del formato 1
    if "result" in df.columns:
        result_df["AI_answer"] = df["result"]
    else:
        result_df["AI_answer"] = ""
    
    # Imposta valori di default per colonne mancanti
    result_df["test_category"] = "UNKNOWN"
    result_df["failed_attempts"] = 0
    
    # Se execution_type non è presente, inferiscilo dal filename
    if "execution_type" not in df.columns or result_df["execution_type"].isna().all():
        inferred_type = infer_execution_type_from_filename(filename)
        result_df["execution_type"] = inferred_type
        logger.info(f"Execution_type inferito dal filename: {inferred_type}")
    
    # Inferisci test_category usando la nuova logica migliorata
    logger.info("Inferendo test_category usando logica migliorata per ORDER BY...")
    
    def infer_category_enhanced(row):
        # Prima prova con la query SQL se disponibile
        if pd.notna(row.get('SQL_query')):
            category = categorize_sql_query(str(row['SQL_query']))
            if category != "UNKNOWN":
                return category
        
        # Se non trovata categoria dalla SQL, prova con il linguaggio naturale
        if pd.notna(row.get('query')):
            return infer_category_from_natural_language(str(row['query']))
        
        return "UNKNOWN"
    
    result_df["test_category"] = result_df.apply(infer_category_enhanced, axis=1)
    
    return result_df

def convert_format_2_to_target(df: pd.DataFrame, filename: str = "") -> pd.DataFrame:
    """
    Converte un DataFrame dal formato 2 al formato target.
    """
    logger.info("Convertendo dal formato 2 al formato target...")
    
    # Il formato 2 è già molto simile al formato target
    result_df = pd.DataFrame(columns=TARGET_COLUMNS)
    
    # Copia tutte le colonne disponibili
    for col in TARGET_COLUMNS:
        if col in df.columns:
            result_df[col] = df[col]
        else:
            # Valori di default per colonne mancanti
            default_values = {
                "test_category": "UNKNOWN",
                "AI_answer": "",
                "failed_attempts": 0,
                "execution_type": "normal",
                "valid_efficiency_score": 0.0,
                "cell_precision": 0.0,
                "cell_recall": 0.0,
                "execution_accuracy": 0,
                "tuple_cardinality": 0.0,
                "tuple_constraint": 0.0,
                "tuple_order": 0.0
            }
            result_df[col] = default_values.get(col, "")
    
    # Se execution_type non è presente, inferiscilo dal filename
    if "execution_type" not in df.columns or result_df["execution_type"].isna().all():
        inferred_type = infer_execution_type_from_filename(filename)
        result_df["execution_type"] = inferred_type
        logger.info(f"Execution_type inferito dal filename: {inferred_type}")
    
    # Migliora le categorie mancanti o di base usando la nuova logica
    logger.info("Migliorando categorizzazione test_category...")
    
    def improve_category(row):
        current_category = row.get('test_category', '')
        
        # Migliora solo se la categoria è mancante, UNKNOWN, ORDERBY, o BASIC
        if pd.isna(current_category) or current_category in ['', 'UNKNOWN', 'ORDERBY', 'BASIC']:
            # Prima prova con la query SQL
            if pd.notna(row.get('SQL_query')):
                new_category = categorize_sql_query(str(row['SQL_query']))
                if new_category != "UNKNOWN":
                    # Se la categoria corrente è ORDERBY o BASIC, aggiorna solo con pattern specifici
                    if current_category in ['ORDERBY', 'BASIC']:
                        # Aggiorna solo se abbiamo trovato un pattern ORDER BY specifico
                        if new_category in ['ORDERBY-SINGLE', 'ORDERBY-ADVANCED', 'ORDERBY-PROJECT']:
                            return new_category
                        else:
                            # Mantieni la categoria originale se non è un pattern ORDER BY specifico
                            return current_category
                    else:
                        # Per UNKNOWN o missing, usa qualsiasi categoria trovata
                        return new_category
            
            # Se non trovata categoria dalla SQL, prova con il linguaggio naturale
            if pd.notna(row.get('query')):
                new_category = infer_category_from_natural_language(str(row['query']))
                if current_category in ['ORDERBY', 'BASIC']:
                    # Aggiorna solo se abbiamo trovato un pattern ORDER BY specifico
                    if new_category in ['ORDERBY-SINGLE', 'ORDERBY-ADVANCED', 'ORDERBY-PROJECT']:
                        return new_category
                    else:
                        # Mantieni la categoria originale
                        return current_category
                else:
                    return new_category
        
        # Se la categoria esistente è OK, mantienila
        return current_category if pd.notna(current_category) else "UNKNOWN"
    
    result_df["test_category"] = result_df.apply(improve_category, axis=1)
    
    return result_df

def clean_and_validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pulisce e valida i dati convertiti.
    """
    logger.info("Pulizia e validazione dati...")
    
    # Rimuovi righe completamente vuote
    df = df.dropna(how='all')
    
    # Pulisci spazi bianchi dalle colonne stringa
    string_columns = ['db_path', 'table_name', 'test_category', 'query', 'model', 'SQL_query', 'execution_type']
    for col in string_columns:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    
    # Converti colonne numeriche
    numeric_columns = [
        'failed_attempts', 'valid_efficiency_score', 'cell_precision', 
        'cell_recall', 'execution_accuracy', 'tuple_cardinality', 
        'tuple_constraint', 'tuple_order'
    ]
    
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
    
    # Valida che le righe abbiano almeno le colonne essenziali
    essential_columns = ['db_path', 'query', 'model']  # Rimosso table_name perché verrà riempito dopo
    before_count = len(df)
    df = df.dropna(subset=essential_columns, how='any')
    after_count = len(df)
    
    if before_count != after_count:
        logger.warning(f"Rimosse {before_count - after_count} righe con dati essenziali mancanti")
    
    # Assicurati che execution_type sia valido
    valid_execution_types = set(CATEGORIES)
    df['execution_type'] = df['execution_type'].apply(
        lambda x: x if x in valid_execution_types else 'normal'
    )
    
    # NUOVO: Riempi le informazioni mancanti delle tabelle
    df = fill_missing_table_info(df)
    
    return df

def separate_by_execution_type(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Separa il DataFrame per execution_type.
    
    Returns:
        Dict con execution_type come chiave e DataFrame corrispondente come valore
    """
    logger.info("Separando dati per execution_type...")
    
    separated_data = {}
    
    if 'execution_type' not in df.columns:
        logger.warning("Colonna execution_type non trovata, tutti i dati andranno in 'normal'")
        separated_data['normal'] = df.copy()
        return separated_data
    
    # Separa per ogni categoria definita
    for category in CATEGORIES:
        category_data = df[df['execution_type'] == category].copy()
        if not category_data.empty:
            # Rimuovi la colonna source_file se presente per il file finale
            if 'source_file' in category_data.columns:
                category_data = category_data.drop('source_file', axis=1)
            separated_data[category] = category_data
            logger.info(f"Categoria '{category}': {len(category_data)} righe")
        else:
            logger.info(f"Categoria '{category}': 0 righe (saltata)")
    
    # Gestisci execution_type non riconosciuti
    unrecognized = df[~df['execution_type'].isin(CATEGORIES)].copy()
    if not unrecognized.empty:
        logger.warning(f"Trovati {len(unrecognized)} record con execution_type non riconosciuti")
        if 'source_file' in unrecognized.columns:
            unrecognized = unrecognized.drop('source_file', axis=1)
        separated_data['unrecognized'] = unrecognized
    
    return separated_data

def process_csv_files(input_directory: str, output_directory: str, exclude_avg: bool = True) -> None:
    """
    Processa tutti i file CSV nella directory specificata e separa in file diversi.
    
    Args:
        input_directory: Directory contenente i file CSV
        output_directory: Directory per i file di output separati
        exclude_avg: Se escludere file con 'avg' nel nome
    """
    logger.info(f"Processando file CSV da: {input_directory}")
    
    # Crea directory di output se non esiste
    os.makedirs(output_directory, exist_ok=True)
    
     # Trova tutti i file CSV ricorsivamente
    csv_pattern = os.path.join(input_directory, "**", "*.csv")
    csv_files = get_all_csv_files(input_directory)
    
    if exclude_avg:
        csv_files = [f for f in csv_files if 'avg' not in os.path.basename(f).lower()]
        logger.info(f"Escludendo file con 'avg' nel nome")
    
    if not csv_files:
        logger.error("Nessun file CSV trovato nella directory specificata")
        return
    
    logger.info(f"Trovati {len(csv_files)} file CSV da processare")
    
    # Lista per raccogliere tutti i DataFrame
    all_dataframes = []
    file_stats = {
        'format_1': 0,
        'format_2': 0,
        'unknown': 0,
        'errors': 0,
        'total_rows': 0
    }
    
    # Processa ogni file
    for file_path in csv_files:
        try:
            filename = os.path.basename(file_path)
            logger.info(f"Processando: {filename}")
            
            # Rileva formato
            format_type = detect_csv_format(file_path)
            
         
            if format_type == 0:
                logger.warning(f"Saltando {filename}: formato non riconosciuto")
                file_stats['unknown'] += 1
                continue
            
            # Leggi il file
            df = pd.read_csv(file_path)
            
            if df.empty:
                logger.warning(f"File vuoto: {filename}")
                continue
            
            logger.info(f"  - Righe: {len(df)}, Formato: {format_type}")
            
            # Converti al formato target
            if format_type == 1:
                converted_df = convert_format_1_to_target(df, filename)
                file_stats['format_1'] += 1
            else:
                converted_df = convert_format_2_to_target(df, filename)
                file_stats['format_2'] += 1
            
            # Aggiungi colonna source_file per tracciabilità
            converted_df['source_file'] = filename
            
            # Pulisci e valida (include il riempimento delle tabelle)
            converted_df = clean_and_validate_data(converted_df)
            
            if not converted_df.empty:
                all_dataframes.append(converted_df)
                file_stats['total_rows'] += len(converted_df)
            else:
                logger.warning(f"Nessuna riga valida trovata in {filename}")
                
        except Exception as e:
            logger.error(f"Errore processando {filename}: {e}")
            file_stats['errors'] += 1
            continue
    
    # Combina tutti i DataFrame
    if not all_dataframes:
        logger.error("Nessun DataFrame valido da combinare")
        return
    
    logger.info("Combinando tutti i DataFrame...")
    combined_df = pd.concat(all_dataframes, ignore_index=True, sort=False)
    
    # Riordina le colonne secondo il formato target
    final_columns = TARGET_COLUMNS + ['source_file']
    combined_df = combined_df[final_columns]
    
    # Separa i dati per execution_type
    separated_data = separate_by_execution_type(combined_df)
    
    # Salva i file separati
    output_files = {}
    for category, category_df in separated_data.items():
        if not category_df.empty:
            output_file = os.path.join(output_directory, f"{category}.csv")
            category_df.to_csv(output_file, index=False)
            output_files[category] = output_file
            logger.info(f"Salvato {category}: {len(category_df)} righe → {output_file}")
    
    # Salva anche il file combinato completo (rimuovendo source_file)
    combined_output_df = combined_df.drop('source_file', axis=1)
    combined_output = os.path.join(output_directory, "all_combined.csv")
    combined_output_df.to_csv(combined_output, index=False)
    logger.info(f"Salvato file combinato completo: {len(combined_output_df)} righe → {combined_output}")

    total_files = len(csv_files)
    processed_files = 0
    skipped_files = 0

    for file_path in csv_files:
        format_type = detect_csv_format(file_path)
        if format_type == 0:
            skipped_files += 1
            logging.warning(f"❌ File saltato (formato sconosciuto): {file_path}")
            continue
        processed_files += 1
        ...
        
    logging.info("STATISTICHE FINALI:")
    logging.info(f"Totale file trovati: {total_files}")
    logging.info(f"File processati con successo: {processed_files}")
    logging.info(f"File saltati: {skipped_files}")
    
    # Stampa statistiche finali
    logger.info("\n" + "="*50)
    logger.info("STATISTICHE FINALI:")
    logger.info(f"File processati con successo: {file_stats['format_1'] + file_stats['format_2']}")
    logger.info(f"  - Formato 1: {file_stats['format_1']}")
    logger.info(f"  - Formato 2: {file_stats['format_2']}")
    logger.info(f"File con formato sconosciuto: {file_stats['unknown']}")
    logger.info(f"File con errori: {file_stats['errors']}")
    logger.info(f"Righe totali elaborate: {len(combined_df)}")
    
    logger.info(f"\nFILE DI OUTPUT CREATI:")
    for category, filepath in output_files.items():
        rows_count = len(separated_data[category])
        logger.info(f"  - {category}.csv: {rows_count} righe")
    logger.info(f"  - all_combined.csv: {len(combined_output_df)} righe (completo)")
    
    # Mostra distribuzione per execution_type
    if 'execution_type' in combined_df.columns:
        exec_type_counts = combined_df['execution_type'].value_counts()
        logger.info(f"\nDistribuzione per execution_type:")
        for exec_type, count in exec_type_counts.items():
            logger.info(f"  {exec_type}: {count}")
    
    # Mostra distribuzione per modelli
    if 'model' in combined_df.columns:
        model_counts = combined_df['model'].value_counts()
        logger.info(f"\nDistribuzione per modello (top 10):")
        for model, count in model_counts.head(10).items():
            logger.info(f"  {model}: {count}")
    
    # Statistiche sull'estrazione delle tabelle
    if 'table_name' in combined_df.columns:
        filled_tables = combined_df['table_name'].notna() & (combined_df['table_name'] != '') & (combined_df['table_name'] != 'nan')
        logger.info(f"\nStatistiche tabelle:")
        logger.info(f"  - Record con tabelle identificate: {filled_tables.sum()}")
        logger.info(f"  - Record senza tabelle: {len(combined_df) - filled_tables.sum()}")
    
    # Statistiche sulle categorie di test migliorate
    if 'test_category' in combined_df.columns:
        category_counts = combined_df['test_category'].value_counts()
        logger.info(f"\nDistribuzione test_category (con nuova logica ORDER BY):")
        for category, count in category_counts.head(15).items():
            logger.info(f"  {category}: {count}")
        
        # Statistiche specifiche per le nuove categorie ORDER BY
        orderby_categories = ['ORDERBY-SINGLE', 'ORDERBY-ADVANCED', 'ORDERBY-PROJECT']
        orderby_total = sum(category_counts.get(cat, 0) for cat in orderby_categories)
        if orderby_total > 0:
            logger.info(f"\nStatistiche ORDER BY dettagliate:")
            for cat in orderby_categories:
                count = category_counts.get(cat, 0)
                if count > 0:
                    percentage = (count / orderby_total) * 100
                    logger.info(f"  {cat}: {count} ({percentage:.1f}% delle ORDER BY)")
    
    logger.info("="*50)
    logger.info("✅ Processo completato con successo!")
    logger.info(f"📁 Tutti i file sono stati salvati in: {output_directory}")

def main():
    """Funzione principale con interfaccia CLI"""
    parser = argparse.ArgumentParser(
        description="Unisce file CSV con formati diversi e separa per execution_type. Include estrazione automatica delle tabelle e logica migliorata per ORDER BY."
    )
    parser.add_argument(
        "input_directory", 
        help="Directory contenente i file CSV da processare"
    )
    parser.add_argument(
        "-o", "--output-dir", 
        default="separated_results",
        help="Directory di output per i file separati (default: separated_results)"
    )
    parser.add_argument(
        "--include-avg", 
        action="store_true",
        help="Includi anche file con 'avg' nel nome (default: esclusi)"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Output verboso"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Verifica che la directory input esista
    if not os.path.isdir(args.input_directory):
        logger.error(f"Directory non trovata: {args.input_directory}")
        return 1
    
    try:
        process_csv_files(
            input_directory=args.input_directory,
            output_directory=args.output_dir,
            exclude_avg=not args.include_avg
        )
        return 0
    except Exception as e:
        logger.error(f"Errore durante l'esecuzione: {e}")
        return 1


if __name__ == "__main__":
    exit(main())