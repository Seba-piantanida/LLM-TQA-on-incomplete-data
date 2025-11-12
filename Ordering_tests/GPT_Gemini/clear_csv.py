import pandas as pd
import os

# === CONFIGURAZIONE ===
cartella_csv = "output/type_fix/categorical_simple_eng_clean"  # Inserisci la tua cartella
csv_riferimento = "output/type_fix/categorical_simple_eng/categorical_simple_eng_type_fix_NORMAL_gpt-5_mini.csv"  # CSV di riferimento

# Carica il CSV di riferimento
df_ref = pd.read_csv(csv_riferimento, dtype=str)
# Mantieni solo le colonne necessarie
df_ref = df_ref[['table', 'sql_query', 'question']]

# Crea un set di tuple per il confronto
ref_set = set(df_ref.itertuples(index=False, name=None))

# Scorri tutti i CSV nella cartella
for file in os.listdir(cartella_csv):
    if file.endswith(".csv"):
        path_file = os.path.join(cartella_csv, file)
        print(f"Processo: {path_file}")

        # Carica il CSV corrente
        df = pd.read_csv(path_file, dtype=str)

        # Se mancano le colonne, salta
        if not all(col in df.columns for col in ['table', 'sql_query', 'question']):
            print(f"⚠️ Il file {file} non contiene tutte le colonne richieste, saltato.")
            continue

        # Filtra le righe che sono nel riferimento
        df_filtered = df[df.apply(lambda row: (row['table'], row['sql_query'], row['question']) in ref_set, axis=1)]

        # Sovrascrivi il file (o salva in una nuova cartella se preferisci)
        df_filtered.to_csv(path_file, index=False)
        print(f"✅ Salvato: {path_file} ({len(df_filtered)} righe)")