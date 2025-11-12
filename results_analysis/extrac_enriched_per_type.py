import pandas as pd
import os

# ============================================================================
# SEZIONE DI CONFIGURAZIONE
# ============================================================================

# Path del file principale A (con colonne extra)
FILE_A_PATH = "enriched_results/FILTRATI/all_combined_enriched.csv"

# Lista di path per i file B, C, D
OTHER_FILES_PATHS = [
    "aggregatetd_results/FILTRATI/normal.csv",
    "aggregatetd_results/FILTRATI/null.csv",
    "aggregatetd_results/FILTRATI/remove.csv"
]

# Colonne comuni presenti in tutti i file
COMMON_COLUMNS = [
    'db_path',
    'table_name',
    'test_category',
    'query',
    'model',
    'AI_answer',
    'SQL_query',
    'failed_attempts',
    'execution_type',
    'valid_efficiency_score',
    'cell_precision',
    'cell_recall',
    'execution_accuracy',
    'tuple_cardinality',
    'tuple_constraint',
    'tuple_order'
]

# ============================================================================
# FUNZIONI PRINCIPALI
# ============================================================================

def read_csv_safe(filepath):
    """Legge un file CSV gestendo eventuali errori."""
    try:
        df = pd.read_csv(filepath, sep=',')
        print(f"✓ Letto file: {filepath} ({len(df)} righe)")
        return df
    except Exception as e:
        print(f"✗ Errore nella lettura di {filepath}: {e}")
        return None

def find_matching_rows(df_main, df_other, common_cols):
    """
    Trova le righe di df_main che matchano con df_other
    basandosi sulle colonne comuni.
    """
    # Crea una chiave unica concatenando i valori delle colonne comuni
    df_main_copy = df_main.copy()
    df_other_copy = df_other.copy()
    
    # Converti i valori in stringhe per il confronto
    df_main_copy['_match_key'] = df_main_copy[common_cols].astype(str).agg('||'.join, axis=1)
    df_other_copy['_match_key'] = df_other_copy[common_cols].astype(str).agg('||'.join, axis=1)
    
    # Trova le chiavi presenti in df_other
    matching_keys = set(df_other_copy['_match_key'])
    
    # Filtra df_main per le righe con chiavi matching
    matched_df = df_main_copy[df_main_copy['_match_key'].isin(matching_keys)].copy()
    
    # Rimuovi la colonna ausiliaria
    matched_df = matched_df.drop(columns=['_match_key'])
    
    return matched_df

def get_output_filename(input_path):
    """
    Genera il nome del file di output aggiungendo _enriched prima dell'estensione.
    """
    directory = os.path.dirname(input_path)
    filename = os.path.basename(input_path)
    name, ext = os.path.splitext(filename)
    output_filename = f"{name}_enriched{ext}"
    
    if directory:
        return os.path.join(directory, output_filename)
    return output_filename

# ============================================================================
# SCRIPT PRINCIPALE
# ============================================================================

def main():
    print("=" * 70)
    print("CSV ENRICHMENT SCRIPT")
    print("=" * 70)
    print()
    
    # Leggi il file A
    print("1. Lettura file principale A...")
    df_a = read_csv_safe(FILE_A_PATH)
    if df_a is None:
        print("Impossibile procedere senza il file A.")
        return
    
    print(f"   Colonne in A: {list(df_a.columns)}")
    print()
    
    # Verifica che le colonne comuni esistano in A
    missing_cols = [col for col in COMMON_COLUMNS if col not in df_a.columns]
    if missing_cols:
        print(f"⚠ Attenzione: colonne comuni mancanti in A: {missing_cols}")
        print()
    
    # Processa ogni file nella lista
    print("2. Processing dei file B, C, D...")
    print()
    
    for i, other_file_path in enumerate(OTHER_FILES_PATHS, 1):
        print(f"   [{i}/{len(OTHER_FILES_PATHS)}] Processing: {other_file_path}")
        
        # Leggi il file
        df_other = read_csv_safe(other_file_path)
        if df_other is None:
            print(f"   → Saltato a causa di errori\n")
            continue
        
        # Verifica colonne comuni
        missing_in_other = [col for col in COMMON_COLUMNS if col not in df_other.columns]
        if missing_in_other:
            print(f"   ⚠ Colonne comuni mancanti in questo file: {missing_in_other}")
        
        # Trova righe matching
        common_cols_present = [col for col in COMMON_COLUMNS if col in df_a.columns and col in df_other.columns]
        matched_df = find_matching_rows(df_a, df_other, common_cols_present)
        
        print(f"   → Trovate {len(matched_df)} righe matching su {len(df_a)} totali in A")
        
        # Genera nome output e salva
        output_path = get_output_filename(other_file_path)
        matched_df.to_csv(output_path, sep=',', index=False)
        print(f"   ✓ Salvato: {output_path}")
        print()
    
    print("=" * 70)
    print("COMPLETATO!")
    print("=" * 70)

if __name__ == "__main__":
    main()