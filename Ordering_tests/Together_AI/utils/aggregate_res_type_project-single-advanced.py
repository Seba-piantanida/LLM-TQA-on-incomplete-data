import pandas as pd
import os
from pathlib import Path

# Definisci le cartelle
input_folder = "output/type_fix"  # MODIFICA QUESTO PATH
output_file_aggregated = "output/type_fix/risultati_aggregated.csv"

# ==================== PARTE 1: Leggi i file dalla cartella ====================
print("Leggendo file dalla cartella...")

# Formato richiesto
required_cols = [
    'db_path', 'table_name', 'test_category', 'query', 'model', 'AI_answer', 
    'SQL_query', 'failed_attempts', 'note', 'valid_efficiency_score', 
    'cell_precision', 'cell_recall', 'execution_accuracy', 
    'tuple_cardinality', 'tuple_constraint', 'tuple_order'
]

input_files = []
# Trova tutti i csv nella cartella e sottocartelle
for file in Path(input_folder).rglob("*.csv"):
    # Escludi file con "summary" nel nome
    if "summary" not in file.name.lower():
        input_files.append(file)

if not input_files:
    print("Nessun file trovato!")
    exit(1)

# Leggi tutti i file
dfs_list = []
for file in input_files:
    try:
        # Leggi il CSV con keep_default_na=False
        df = pd.read_csv(file, keep_default_na=False, na_values=[''])
        
        # Verifica che il file abbia il formato corretto
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"  ⚠ Saltato {file.name}: mancano colonne {missing_cols}")
            continue
        
        # Determina il valore di exec_profile basandosi sul nome del file
        filename_lower = file.name.lower()
        
        if "clean" in filename_lower and "null" in filename_lower:
            exec_profile = "CLEAN_NULL"
        elif "clean" in filename_lower and ("rem" in filename_lower or "remove" in filename_lower):
            exec_profile = "CLEAN_REMOVE"
        elif "null" in filename_lower:
            exec_profile = "FULL_NULL"
        elif "rem" in filename_lower or "remove" in filename_lower:
            exec_profile = "FULL_REMOVE"
        else:
            exec_profile = "FULL_NORMAL"
        
        # Aggiungi la colonna exec_profile
        df['exec_profile'] = exec_profile
        
        dfs_list.append(df)
        print(f"  ✓ Letto: {file.name} → {exec_profile} ({len(df)} righe)")
    except Exception as e:
        print(f"  ✗ Errore leggendo {file.name}: {e}")

if not dfs_list:
    print("Nessun file valido trovato!")
    exit(1)

# Concatena tutti i dataframe
df = pd.concat(dfs_list, ignore_index=True)
print(f"\nDataframe creato con {len(df)} righe")

# ==================== PARTE 2: Rinomina colonne ====================
print("\nRinominando colonne...")

df.rename(columns={
    'table_name': 'table',
    'test_category': 'sql_tag'
}, inplace=True)

print(f"Colonne dopo rinomina: {', '.join(df.columns)}")

# ==================== PARTE 3: Aggregazione ====================
print("\n" + "="*70)
print("AGGREGAZIONE DEI DATI")
print("="*70)

# Colonne per il raggruppamento
group_cols = ['model', 'table', 'sql_tag', 'exec_profile']

# Colonne numeriche da aggregare (media)
numeric_cols = [
    'valid_efficiency_score', 'cell_precision', 'cell_recall', 
    'execution_accuracy', 'tuple_cardinality', 'tuple_constraint', 'tuple_order'
]

# Verifica quali colonne numeriche sono effettivamente presenti
available_numeric_cols = [col for col in numeric_cols if col in df.columns]
missing_numeric_cols = [col for col in numeric_cols if col not in df.columns]

if missing_numeric_cols:
    print(f"\n⚠ Colonne numeriche non trovate (verranno ignorate): {', '.join(missing_numeric_cols)}")

if not available_numeric_cols:
    print("\n✗ ERRORE: Nessuna colonna numerica disponibile per l'aggregazione!")
    exit(1)

print(f"\nColonne numeriche da aggregare: {', '.join(available_numeric_cols)}")

# Verifica che tutte le colonne di raggruppamento esistano
missing_group_cols = [col for col in group_cols if col not in df.columns]
if missing_group_cols:
    print(f"\n✗ ERRORE: Colonne di raggruppamento mancanti: {', '.join(missing_group_cols)}")
    exit(1)

# Converti le colonne numeriche in tipo numerico (gestendo eventuali stringhe)
for col in available_numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Esegui l'aggregazione
print(f"\nRaggruppando per: {', '.join(group_cols)}")
df_aggregated = df.groupby(group_cols, as_index=False)[available_numeric_cols].mean()

# Aggiungi una colonna con il conteggio delle righe per gruppo
df_count = df.groupby(group_cols, as_index=False).size()
df_count.rename(columns={'size': 'num_queries'}, inplace=True)

# Unisci il conteggio con i dati aggregati
df_aggregated = df_aggregated.merge(df_count, on=group_cols, how='left')

# Riordina le colonne nel formato richiesto
final_cols = ['model', 'table', 'sql_tag', 'exec_profile'] + available_numeric_cols + ['num_queries']
df_aggregated = df_aggregated[final_cols]

print(f"\nDataframe aggregato creato con {len(df_aggregated)} righe")
print(f"Colonne finali: {', '.join(df_aggregated.columns)}")

# Salva il dataframe aggregato
df_aggregated.to_csv(output_file_aggregated, index=False)
print(f"\n✓ File aggregato salvato: {output_file_aggregated}")

# Mostra statistiche
print("\nStatistiche aggregazione:")
print(f"  - Numero totale di gruppi: {len(df_aggregated)}")
print(f"  - Modelli unici: {df_aggregated['model'].nunique()}")
print(f"  - Tabelle uniche: {df_aggregated['table'].nunique()}")
print(f"  - SQL tags unici: {df_aggregated['sql_tag'].nunique()}")
print(f"  - Profili di esecuzione unici: {df_aggregated['exec_profile'].nunique()}")

# Mostra un sample dei dati aggregati
print("\nPrime 5 righe del dataframe aggregato:")
print(df_aggregated.head(5).to_string())