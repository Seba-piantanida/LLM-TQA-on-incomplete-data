import pandas as pd

def filtra_csv():
    # Chiedi all'utente il percorso del file CSV
    input_file = input("Inserisci il nome del file CSV (es. dati.csv): ").strip()

    try:
        # Leggi il file CSV
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"⚠️ File '{input_file}' non trovato.")
        return
    except Exception as e:
        print(f"⚠️ Errore nella lettura del file: {e}")
        return

    # Mostra le colonne disponibili
    print("\nColonne disponibili:")
    print(", ".join(df.columns))

    # Chiedi all'utente la colonna e il valore da filtrare
    colonna = input("\nInserisci il nome della colonna da filtrare: ").strip()
    
    if colonna not in df.columns:
        print(f"⚠️ La colonna '{colonna}' non esiste nel file.")
        return

    valore = input(f"Inserisci il valore da cercare nella colonna '{colonna}': ").strip()

    # Filtra il DataFrame
    df_filtrato = df[df[colonna].astype(str) == valore]

    # Nome del file di output
    output_file = f"filtrato_gemini_pro.csv"

    # Salva il file CSV filtrato
    df_filtrato.to_csv(output_file, index=False)

    print(f"\n✅ File salvato come '{output_file}' con {len(df_filtrato)} righe.")

if __name__ == "__main__":
    filtra_csv()
