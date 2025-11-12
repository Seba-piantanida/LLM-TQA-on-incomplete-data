import pandas as pd
import os

# Configura i percorsi
input_csv = "tests/test_music_join_cut_300.csv"  # CSV originale

# Array con i nuovi valori di db_path
db_paths = [

    "data/db_music.sqlite",
    
    
]

def main():
    # Leggi il CSV originale
    df = pd.read_csv(input_csv)

    # Nome base del file di input (senza estensione)
    base_name, ext = os.path.splitext(input_csv)

    # Genera un file per ogni valore di db_path
    for new_path in db_paths:
        df_modified = df.copy()
        df_modified["db_path"] = new_path  # sostituzione

        # prendo solo il nome del db senza cartelle (es: db1.sqlite)
        db_name = os.path.basename(new_path)

        # costruiamo nome output: input + "_" + db_name.csv
        output_file = f"{base_name}_{db_name}"

        df_modified.to_csv(output_file, index=False)
        print(f"Creato: {output_file} con db_path={new_path}")

if __name__ == "__main__":
    main()
