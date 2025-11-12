import os
import pandas as pd
import sqlite3


def csv_to_sqlite(csv_file: str, db_file: str, table_name: str):
    """Importa un CSV in una tabella SQLite, creando il DB se non esiste."""
    
    # Controllo CSV
    if not os.path.exists(csv_file):
        print(f"❌ File CSV non trovato: {csv_file}")
        return

    # Se il DB non esiste, verrà creato automaticamente
    db_exists = os.path.exists(db_file)

    # Connessione al DB
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    # Log creazione
    if not db_exists:
        print(f"📂 Database '{db_file}' non trovato. Creazione in corso...")
    else:
        print(f"✅ Database '{db_file}' trovato.")

    # Lettura CSV
    df = pd.read_csv(csv_file)

    # Creazione o sostituzione tabella
    df.to_sql(table_name, conn, if_exists='replace', index=False)

    conn.commit()
    conn.close()

    print(f"📥 Importato {len(df)} record da '{csv_file}' nella tabella '{table_name}'.\n")


if __name__ == "__main__":
    DB_FILE = "data/categorical_simple_eng.sqlite"

    csv_to_sqlite("data/categorical_simple_eng/animals_eng.csv", DB_FILE, "animals")
    csv_to_sqlite("data/categorical_simple_eng/cars_eng.csv", DB_FILE, "cars")
    csv_to_sqlite("data/categorical_simple_eng/monuments.csv", DB_FILE, "monuments")
    csv_to_sqlite("data/categorical_simple_eng/movies_eng.csv", DB_FILE, "movies")