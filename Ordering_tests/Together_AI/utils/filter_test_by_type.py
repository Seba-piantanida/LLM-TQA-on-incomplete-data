import os
import pandas as pd

# Cartella di partenza
cartella_base = "tests/advanced_num_only"

# Valore da filtrare
valore_filtro = "ORDERBY-ADVANCED"

for root, dirs, files in os.walk(cartella_base):
    for file in files:
        if file.lower().endswith(".csv"):
            percorso_file = os.path.join(root, file)
            try:
                # Leggi CSV
                df = pd.read_csv(percorso_file)
                
                # Controlla che esista la colonna
                if "sql_tag" in df.columns:
                    # Filtra le righe
                    df_filtrato = df[df["sql_tag"] == valore_filtro]
                    
                    # Sovrascrivi il file
                    df_filtrato.to_csv(percorso_file, index=False)
                    print(f"File filtrato: {percorso_file}")
                else:
                    print(f"Colonna 'sql_tag' mancante in: {percorso_file}")
            except Exception as e:
                print(f"Errore con {percorso_file}: {e}")