#!/usr/bin/env python3
"""
Script per unire file CSV con due formati diversi in un formato standardizzato.
Prende tutti i CSV in una directory (escludendo quelli con 'avg' nel nome)
e li converte nel formato target standardizzato, poi li separa per categoria.
"""

import pandas as pd
import os
import glob
import logging
import re
from pathlib import Path
from typing import List, Dict, Optional
import argparse

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

def infer_category_from_sql(sql_query: str) -> str:
    """
    Inferisce la categoria test_category basandosi sulla query SQL.
    """
    if pd.isna(sql_query) or not sql_query:
        return "UNKNOWN"
    
    # Normalizza la query: rimuovi spazi extra e converti in minuscolo
    query = re.sub(r'\s+', ' ', str(sql_query).strip().upper())
    
    # Controlla se ha ORDER BY e LIMIT
    has_order_by = bool(re.search(r'\bORDER\s+BY\b', query))
    has_limit = bool(re.search(r'\bLIMIT\s+\d+', query))
    
    if has_order_by and has_limit:
        # Estrai la parte SELECT per verificare se è SELECT *
        select_match = re.search(r'SELECT\s+(.*?)\s+FROM', query)
        if select_match:
            select_part = select_match.group(1).strip()
            
            # Controlla se è SELECT *
            if select_part == '*':
                # Estrai la parte ORDER BY per contare le colonne
                order_by_match = re.search(r'ORDER\s+BY\s+(.*?)(?:\s+LIMIT|\s*$)', query)
                if order_by_match:
                    order_by_part = order_by_match.group(1).strip()
                    
                    # Rimuovi ASC/DESC per contare le colonne
                    order_by_clean = re.sub(r'\b(ASC|DESC)\b', '', order_by_part)
                    # Conta le virgole per determinare il numero di colonne
                    num_columns = len([col.strip() for col in order_by_clean.split(',') if col.strip()])
                    
                    if num_columns == 1:
                        return "ORDERBY-SINGLE"
                    else:
                        return "ORDERBY-ADVANCED"
            else:
                # È una projection (SELECT specifiche colonne)
                return "ORDERBY-PROJECT"
    
    # Fallback alla logica originale per altri tipi di query
    query_lower = query.lower()
    
    if "order by" in query_lower and "limit" in query_lower:
        return "ORDERBY-LIMIT"  # Fallback generico
    elif "order by" in query_lower:
        return "ORDERBY"
    elif "group by" in query_lower:
        return "GROUPBY"
    elif "join" in query_lower:
        return "JOIN"
    elif "where" in query_lower:
        return "FILTER"
    elif "sum" in query_lower or "avg" in query_lower or "count" in query_lower:
        return "AGGREGATE"
    elif "distinct" in query_lower:
        return "DISTINCT"
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
    
    # Inferisci test_category dalla SQL_query usando la nuova logica
    if "SQL_query" in result_df.columns:
        result_df["test_category"] = result_df["SQL_query"].apply(infer_category_from_sql)
        logger.info("Test_category inferito dalle query SQL usando la logica aggiornata")
    
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
    
    # Se test_category è mancante o UNKNOWN, inferiscilo dalla SQL_query
    if "test_category" not in df.columns or result_df["test_category"].isna().any() or (result_df["test_category"] == "UNKNOWN").any():
        if "SQL_query" in result_df.columns:
            # Applica la nuova logica solo alle righe con test_category mancante o UNKNOWN
            mask = result_df["test_category"].isna() | (result_df["test_category"] == "UNKNOWN")
            result_df.loc[mask, "test_category"] = result_df.loc[mask, "SQL_query"].apply(infer_category_from_sql)
            logger.info("Test_category inferito dalle query SQL per righe mancanti usando la logica aggiornata")
    
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
    essential_columns = ['db_path', 'table_name', 'query', 'model']
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
    csv_files = glob.glob(csv_pattern, recursive=True)
    
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
            
            # Pulisci e valida
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
    
    # Mostra distribuzione per test_category
    if 'test_category' in combined_df.columns:
        category_counts = combined_df['test_category'].value_counts()
        logger.info(f"\nDistribuzione per test_category:")
        for test_cat, count in category_counts.items():
            logger.info(f"  {test_cat}: {count}")
    
    # Mostra distribuzione per modelli
    if 'model' in combined_df.columns:
        model_counts = combined_df['model'].value_counts()
        logger.info(f"\nDistribuzione per modello (top 10):")
        for model, count in model_counts.head(10).items():
            logger.info(f"  {model}: {count}")
    
    logger.info("="*50)
    logger.info("✅ Processo completato con successo!")
    logger.info(f"📁 Tutti i file sono stati salvati in: {output_directory}")

def main():
    """Funzione principale con interfaccia CLI"""
    parser = argparse.ArgumentParser(
        description="Unisce file CSV con formati diversi e separa per execution_type"
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