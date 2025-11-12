import pandas as pd
import os
from pathlib import Path

# Definisci le cartelle
test_folder = "tests/single_table_only"
risultati_folder = "results/type_fix/single_table_only"
output_file = "results/type_fix/single_table_only/risultati_merged.csv"
output_file_aggregated = "results/type_fix/single_table_only/risultati_aggregated.csv"

# ==================== PARTE 1: Leggi i file dalla cartella test ====================
print("Leggendo file dalla cartella test...")

test_files = []
for file in Path(test_folder).glob("*.csv"):
    # Salta i file che contengono "clean" nel nome
    if "clean" not in file.name.lower():
        test_files.append(file)

if not test_files:
    print("Nessun file valido trovato nella cartella test!")
    exit(1)

# Leggi e unisci tutti i file in un unico dataframe A
dfs_test = []
for file in test_files:
    try:
        # Leggi il CSV con keep_default_na=False per evitare che stringhe vengano interpretate come NaN
        df = pd.read_csv(file, keep_default_na=False, na_values=[''])
        
        # Verifica che le colonne richieste esistano
        required_cols = ['db_path', 'db_id', 'tbl_name', 'test_category', 'sql_tag', 'query', 'question']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"  ⚠ Saltato {file.name}: mancano colonne {missing_cols}")
            continue
        
        # Mantieni solo le colonne richieste
        df = df[required_cols]
        dfs_test.append(df)
        print(f"  ✓ Letto: {file.name} ({len(df)} righe)")
    except Exception as e:
        print(f"  ✗ Errore leggendo {file.name}: {e}")

if not dfs_test:
    print("Nessun file valido con le colonne corrette trovato nella cartella test!")
    exit(1)

df_A = pd.concat(dfs_test, ignore_index=True)
print(f"\nDataframe A creato con {len(df_A)} righe")
print(f"Colonne: {', '.join(df_A.columns)}")
print(f"Valori unici in sql_tag (primi 10): {df_A['sql_tag'].unique()[:10].tolist()}")

# ==================== PARTE 2: Leggi i file dalla cartella risultati ====================
print(f"\nLeggendo file dalla cartella risultati e sottocartelle...")

# Colonne minime richieste per i file risultati
minimum_required_cols = ['model', 'execution_type', 'db_path', 'table', 'sql_query', 'question']

risultati_files = []
# Trova tutti i csv nella cartella risultati e sottocartelle
for file in Path(risultati_folder).rglob("*.csv"):
    risultati_files.append(file)

if not risultati_files:
    print("Nessun file trovato nella cartella risultati!")
    exit(1)

# Leggi tutti i file e aggiungi la colonna exec_profile
dfs_risultati = []
for file in risultati_files:
    try:
        # Leggi il CSV con keep_default_na=False per evitare che stringhe vengano interpretate come NaN
        df = pd.read_csv(file, keep_default_na=False, na_values=[''])
        
        # Verifica che il file abbia almeno le colonne minime richieste
        missing_cols = [col for col in minimum_required_cols if col not in df.columns]
        if missing_cols:
            print(f"  ⚠ Saltato {file.relative_to(risultati_folder)}: mancano colonne minime {missing_cols}")
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
        
        dfs_risultati.append(df)
        print(f"  ✓ Letto: {file.relative_to(risultati_folder)} → {exec_profile} ({len(df)} righe, {len(df.columns)} colonne)")
    except Exception as e:
        print(f"  ✗ Errore leggendo {file.name}: {e}")

if not dfs_risultati:
    print("Nessun file valido con le colonne minime richieste trovato nella cartella risultati!")
    exit(1)

df_B = pd.concat(dfs_risultati, ignore_index=True)
print(f"\nDataframe B creato con {len(df_B)} righe")
print(f"Colonne: {', '.join(df_B.columns)}")

# ==================== PARTE 3: Join per aggiungere sql_tag ====================
print("\nEffettuando il join tra df_A e df_B...")

# Rimuovi spazi bianchi dalle colonne question per un join più robusto
df_A['question'] = df_A['question'].astype(str).str.strip()
df_B['question'] = df_B['question'].astype(str).str.strip()

# Verifica che la colonna question esista in entrambi i dataframe
if 'question' not in df_A.columns:
    print("ERRORE: La colonna 'question' non esiste in df_A")
    exit(1)
if 'question' not in df_B.columns:
    print("ERRORE: La colonna 'question' non esiste in df_B")
    exit(1)

# Crea un dataframe temporaneo con solo question e sql_tag da A
df_A_join = df_A[['question', 'sql_tag']].drop_duplicates(subset=['question'])

print(f"Domande uniche in df_A: {len(df_A_join)}")
print(f"Domande uniche in df_B: {df_B['question'].nunique()}")

# Debug: mostra alcuni esempi di sql_tag in df_A
print(f"\nEsempi di sql_tag in df_A:")
print(df_A_join[['question', 'sql_tag']].head(3).to_string())

# Fai il left join per aggiungere sql_tag da A a B
df_B = df_B.merge(
    df_A_join, 
    on='question', 
    how='left'
)

print(f"\nJoin completato. Righe nel dataframe finale: {len(df_B)}")
print(f"Righe con sql_tag trovato: {df_B['sql_tag'].notna().sum()}")
print(f"Righe con sql_tag vuoto: {(df_B['sql_tag'] == '').sum()}")
print(f"Righe senza sql_tag (no match): {df_B['sql_tag'].isna().sum()}")

# Salva il dataframe B finale
df_B.to_csv(output_file, index=False)
print(f"\n✓ File salvato: {output_file}")
print(f"Colonne finali: {', '.join(df_B.columns)}")

# Mostra un sample dei dati
print("\nPrime 5 righe del risultato:")
if 'sql_tag' in df_B.columns:
    cols_to_show = ['question', 'sql_tag', 'exec_profile']
    print(df_B[cols_to_show].head(5).to_string())
else:
    print(df_B.head(5))

# Mostra statistiche sul match
unmatched = df_B['sql_tag'].isna() | (df_B['sql_tag'] == '')
if unmatched.sum() > 0:
    print(f"\n⚠ {unmatched.sum()} domande non hanno trovato corrispondenza. Esempi:")
    print(df_B[unmatched]['question'].head(3).tolist())

# ==================== PARTE 4: Aggregazione ====================
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

# Verifica quali colonne numeriche sono effettivamente presenti nel dataframe
available_numeric_cols = [col for col in numeric_cols if col in df_B.columns]
missing_numeric_cols = [col for col in numeric_cols if col not in df_B.columns]

if missing_numeric_cols:
    print(f"\n⚠ Colonne numeriche non trovate (verranno ignorate): {', '.join(missing_numeric_cols)}")

if not available_numeric_cols:
    print("\n✗ ERRORE: Nessuna colonna numerica disponibile per l'aggregazione!")
    exit(1)

print(f"\nColonne numeriche da aggregare: {', '.join(available_numeric_cols)}")

# Verifica che tutte le colonne di raggruppamento esistano
missing_group_cols = [col for col in group_cols if col not in df_B.columns]
if missing_group_cols:
    print(f"\n✗ ERRORE: Colonne di raggruppamento mancanti: {', '.join(missing_group_cols)}")
    exit(1)

# Converti le colonne numeriche in tipo numerico (gestendo eventuali stringhe)
for col in available_numeric_cols:
    df_B[col] = pd.to_numeric(df_B[col], errors='coerce')

# Esegui l'aggregazione
print(f"\nRaggruppando per: {', '.join(group_cols)}")
df_aggregated = df_B.groupby(group_cols, as_index=False)[available_numeric_cols].mean()

# Aggiungi una colonna con il conteggio delle righe per gruppo
df_count = df_B.groupby(group_cols, as_index=False).size()
df_count.rename(columns={'size': 'num_queries'}, inplace=True)

# Unisci il conteggio con i dati aggregati
df_aggregated = df_aggregated.merge(df_count, on=group_cols, how='left')

print(f"\nDataframe aggregato creato con {len(df_aggregated)} righe")
print(f"Colonne: {', '.join(df_aggregated.columns)}")

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