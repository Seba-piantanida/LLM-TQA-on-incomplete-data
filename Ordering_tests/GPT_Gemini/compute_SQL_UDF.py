import os
import sqlite3
import pandas as pd

# === UDF su date (formato atteso: "dd-mm-yyyy") ===
def extract_day(date_str):
    if not date_str:
        return None
    try:
        return int(date_str.split('-')[0])
    except Exception:
        return None

def extract_month(date_str):
    if not date_str:
        return None
    try:
        return int(date_str.split('-')[1])
    except Exception:
        return None

def extract_year(date_str):
    if not date_str:
        return None
    try:
        return int(date_str.split('-')[2])
    except Exception:
        return None

# === UDF su nomi ===
def extract_first_name(name):
    if not name:
        return None
    return " ".join(name.strip().split()[:-1]) or name.strip().split()[0]

def extract_last_name(name):
    if not name:
        return None
    parts = name.strip().split()
    return parts[-1] if len(parts) > 1 else None

# === UDF extra ===
def extract_initials(name):
    if not name:
        return None
    return ''.join([p[0].upper() for p in name.strip().split() if p])

def reverse_string(s):
    if not s:
        return None
    return s[::-1]

def word_count(s):
    if not s:
        return 0
    return len(s.strip().split())


# === Funzione per registrare tutte le UDF in SQLite ===
def register_udfs(conn):
    conn.create_function("extract_day", 1, extract_day)
    conn.create_function("extract_month", 1, extract_month)
    conn.create_function("extract_year", 1, extract_year)
    conn.create_function("extract_first_name", 1, extract_first_name)
    conn.create_function("extract_last_name", 1, extract_last_name)
    conn.create_function("extract_initials", 1, extract_initials)
    conn.create_function("reverse_string", 1, reverse_string)
    conn.create_function("word_count", 1, word_count)


# === Esegui query e ritorna risultato come liste di liste ===
def execute_query(db_path, query):
    conn = sqlite3.connect(db_path)
    register_udfs(conn)
    cur = conn.cursor()
    try:
        cur.execute(query)
        rows = cur.fetchall()
        # Trasformiamo direttamente in lista di liste (senza chiavi)
        result = [list(row) for row in rows]
    except Exception as e:
        result = f"ERROR: {e}"
    finally:
        conn.close()
    return result


# === Elabora tutti i CSV in una cartella ===
def process_csv_folder(input_dir):
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".csv"):
                file_path = os.path.join(root, file)
                print(f"Processing {file_path} ...")
                df = pd.read_csv(file_path)

                # Controllo che ci siano tutte le colonne richieste
                required_cols = [
                    "model","execution_type","db_path",
                    "table","sql_query","question","result","tables_used"
                ]
                if not all(col in df.columns for col in required_cols):
                    print(f"⚠️  Skip {file_path}: mancano colonne richieste")
                    continue

                new_queries = []
                for _, row in df.iterrows():
                    db_path = row["db_path"]
                    query = row["sql_query"]
                    if pd.isna(query) or pd.isna(db_path):
                        new_queries.append(query)
                        continue
                    res = execute_query(db_path, query)
                    new_queries.append(str(res))  # lista di liste come stringa

                df["sql_query"] = new_queries

                # Percorso output nella sotto-cartella SQL
                rel_path = os.path.relpath(file_path, input_dir)
                out_dir = os.path.join(input_dir, "SQL", os.path.dirname(rel_path))
                os.makedirs(out_dir, exist_ok=True)
                out_file = os.path.join(out_dir, file)

                # Mantiene l'ordine originale delle colonne
                df = df[required_cols]
                df.to_csv(out_file, index=False)
                print(f"✅ File salvato in: {out_file}")


if __name__ == "__main__":
    input_dir = "output/type_fix/music_join"  # cambia con la tua cartella
    process_csv_folder(input_dir)