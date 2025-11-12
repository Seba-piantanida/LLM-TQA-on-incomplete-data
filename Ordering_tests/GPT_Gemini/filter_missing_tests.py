import pandas as pd

# === Percorsi dei file ===
file_a = 'tests/test_numerical_simple.csv'
file_b = 'output/numerical_simple_normal_gpt-4o_mini.csv'
file_c = 'tests/test_numerical_simple_gpt_missing.csv'

# === Caricamento CSV ===
df_a = pd.read_csv(file_a, dtype=str)
df_b = pd.read_csv(file_b, dtype=str)

# === Assicurati che le colonne siano presenti ===
if 'query' not in df_a.columns or 'sql_query' not in df_b.columns:
    raise ValueError("Colonne mancanti: assicurati che A abbia 'Query' e B abbia 'sql_query'.")

# === Crea set dei valori da escludere ===
queries_b = set(df_b['sql_query'].dropna().unique())

# === Filtro: tieni solo righe di A che NON sono in B ===
df_c = df_a[~df_a['query'].isin(queries_b)]

# === Salva risultato ===
df_c.to_csv(file_c, index=False)
print(f"File C salvato con {len(df_c)} righe in: {file_c}")