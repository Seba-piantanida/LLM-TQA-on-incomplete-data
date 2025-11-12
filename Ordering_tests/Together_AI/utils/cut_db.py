import sqlite3
import os

def crea_db_con_limite_per_tabella(original_db_path, max_righe_per_tabella):
    # Verifica che il file originale esista
    if not os.path.exists(original_db_path):
        print(f"Il file {original_db_path} non esiste.")
        return

    # Genera il nome del nuovo DB
    base_name, ext = os.path.splitext(original_db_path)
    nuovo_db_path = f"{base_name}_r_{max_righe_per_tabella}{ext}"

    # Connessioni
    conn_old = sqlite3.connect(original_db_path)
    cur_old = conn_old.cursor()

    conn_new = sqlite3.connect(nuovo_db_path)
    cur_new = conn_new.cursor()

    # Recupera le tabelle escludendo quelle di sistema
    cur_old.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
    tabelle = [row[0] for row in cur_old.fetchall()]

    if not tabelle:
        print("Nessuna tabella trovata.")
        return

    for tabella in tabelle:
        print(f"Processando la tabella: {tabella}")

        # Crea la stessa tabella nel nuovo DB
        cur_old.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{tabella}';")
        create_sql = cur_old.fetchone()[0]
        cur_new.execute(create_sql)

        # Conta righe disponibili
        cur_old.execute(f"SELECT COUNT(*) FROM {tabella};")
        totale = cur_old.fetchone()[0]
        limite = min(totale, max_righe_per_tabella)

        # Estrai righe fino al limite
        if limite > 0:
            cur_old.execute(f"SELECT * FROM {tabella} LIMIT {limite};")
            righe = cur_old.fetchall()

            placeholders = ", ".join(["?"] * len(righe[0]))
            cur_new.executemany(f"INSERT INTO {tabella} VALUES ({placeholders});", righe)

    # Chiudi e salva
    conn_new.commit()
    conn_old.close()
    conn_new.close()
    print(f"Database creato in '{nuovo_db_path}' con massimo {max_righe_per_tabella} righe per tabella.")

# Esempio d'uso:
crea_db_con_limite_per_tabella("data/db_categorical_simple.sqlite", 20)