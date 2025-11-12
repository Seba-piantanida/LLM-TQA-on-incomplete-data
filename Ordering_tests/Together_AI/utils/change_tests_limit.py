import csv
import os
import re

# === INPUT UTENTE ===
input_dir = input("📂 Inserisci il percorso della cartella con i CSV: ").strip()
output_dir = input("💾 Inserisci il percorso della cartella di output: ").strip()
n = int(input("🔢 Inserisci il numero di LIMIT (0 = tutti): ").strip())

# Crea cartella di output se non esiste
os.makedirs(output_dir, exist_ok=True)

def aggiorna_sql(sql, n):
    """Aggiorna il LIMIT nella query SQL."""
    if n == 0:
        return re.sub(r"\s+LIMIT\s+\d+\s*$", "", sql, flags=re.IGNORECASE)
    else:
        if re.search(r"LIMIT\s+\d+", sql, flags=re.IGNORECASE):
            return re.sub(r"(LIMIT\s+)\d+", rf"\g<1>{n}", sql, flags=re.IGNORECASE)
        else:
            return sql.strip() + f" LIMIT {n}"

def aggiorna_frase(frase, n):
    """Aggiorna la frase in linguaggio naturale."""
    # Rimuovi testo tra parentesi tonde
    frase = re.sub(r"\([^)]*\)", "", frase).strip()

    if n == 0:
        # Sostituzioni per 'tutti'
        frase = re.sub(r"\btop\s+\d+\s+rows\b", "all rows", frase, flags=re.IGNORECASE)
        frase = re.sub(r"\btop\s+\d+\s+values\b", "all values", frase, flags=re.IGNORECASE)
        frase = re.sub(r"\bfirst\s+\d+\s+rows\b", "all rows", frase, flags=re.IGNORECASE)
        frase = re.sub(r"\bfirst\s+\d+\s+values\b", "all values", frase, flags=re.IGNORECASE)
        frase = re.sub(r"\bfirst\s+all\s+rows\b", "all rows", frase, flags=re.IGNORECASE)
        frase = re.sub(r"\bfirst\s+all\s+values\b", "all values", frase, flags=re.IGNORECASE)
    else:
        # Sostituzioni con numero specifico
        frase = re.sub(r"\btop\s+\d+\s+rows\b", f"top {n} rows", frase, flags=re.IGNORECASE)
        frase = re.sub(r"\btop\s+\d+\s+values\b", f"top {n} values", frase, flags=re.IGNORECASE)
        frase = re.sub(r"\bfirst\s+\d+\s+rows\b", f"first {n} rows", frase, flags=re.IGNORECASE)
        frase = re.sub(r"\bfirst\s+\d+\s+values\b", f"first {n} values", frase, flags=re.IGNORECASE)

    return re.sub(r"\s+", " ", frase).strip()

def processa_csv(file_csv, output_folder, n):
    """Elabora un singolo CSV."""
    output_path = os.path.join(output_folder, os.path.basename(file_csv))
    with open(file_csv, newline="", encoding="utf-8") as f_in, \
         open(output_path, "w", newline="", encoding="utf-8") as f_out:

        reader = csv.DictReader(f_in)
        writer = csv.DictWriter(f_out, fieldnames=reader.fieldnames)
        writer.writeheader()

        for row in reader:
            row["query"] = aggiorna_sql(row["query"], n)
            row["question"] = aggiorna_frase(row["question"], n)
            writer.writerow(row)

    print(f"✅ Salvato: {output_path}")

# Processa solo i file CSV nella cartella principale
for file in os.listdir(input_dir):
    if file.lower().endswith(".csv"):
        processa_csv(os.path.join(input_dir, file), output_dir, n)

print("🏁 Completato!")