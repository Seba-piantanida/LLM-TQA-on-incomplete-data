import sqlite3
import csv
import os
import glob

def csv_to_sqlite_all(csv_folder, db_file):
    # Crea connessione al DB SQLite
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    # Trova tutti i file CSV nella cartella
    csv_files = glob.glob(os.path.join(csv_folder, '*.csv'))

    for csv_path in csv_files:
        table_name = os.path.splitext(os.path.basename(csv_path))[0]  # Usa il nome del file come nome tabella
        print(f"Inserimento di {csv_path} nella tabella '{table_name}'...")

        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                header = next(reader)

                # Crea la tabella se non esiste
                columns = ', '.join([f'"{col}" TEXT' for col in header])
                create_query = f'CREATE TABLE IF NOT EXISTS "{table_name}" ({columns});'
                cursor.execute(create_query)

                # Prepara la query di inserimento
                placeholders = ', '.join(['?'] * len(header))
                insert_query = f'INSERT INTO "{table_name}" VALUES ({placeholders});'

                # Inserisci ogni riga
                for row in reader:
                    cursor.execute(insert_query, row)

        except Exception as e:
            print(f"❌ Errore nel file {csv_path}: {e}")

    # Commit e chiusura connessione
    conn.commit()
    conn.close()
    print(f"\n✅ Tutti i CSV nella cartella '{csv_folder}' sono stati importati nel DB '{db_file}'.")

# === ESEMPIO DI UTILIZZO ===
csv_folder = 'data/numerical_simple_2'      # cartella con i CSV
db_file = 'data/db_numerical_simple_2.sqlite'  # file .sqlite finale

csv_to_sqlite_all(csv_folder, db_file)