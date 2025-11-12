import os
import pandas as pd
import re
from pathlib import Path

# ✅ CONFIGURAZIONE
CSV_DIR = 'output/complete_cut_300'  # <--- 🔁 Cambia questo
MODELS = ['gpt-4o_mini', 'gemini-2.5_flash']
EXCLUDE_KEYS = ['sum']

def detect_model(filename):
    """Rileva il nome del modello dal nome del file."""
    for model in MODELS:
        if model in filename:
            return model
    return 'unknown'

def extract_ordered_entries(text):
    """
    Estrae manualmente le 'ordered_entries' da una stringa (anche malformata)
    e restituisce una lista di liste.
    """
    try:
        # Cerca il blocco "ordered_entries": [...]
        match = re.search(r'"?ordered_entries"?\s*:\s*\[(.*?)\]\s*}', text, re.DOTALL)
        if not match:
            return text  # Non contiene ordered_entries → restituisce com'è

        entries_block = match.group(1)

        # Estrae singoli dizionari ({}), uno per riga
        entry_texts = re.findall(r'\{(.*?)\}', entries_block, re.DOTALL)
        result = []

        for entry in entry_texts:
            # Trova coppie chiave: valore
            pairs = re.findall(r'"?(\w+)"?\s*:\s*"?(.*?)"?\s*(?:,|$)', entry)
            row = []
            for key, value in pairs:
                if key not in EXCLUDE_KEYS:
                    clean_val = value.strip()
                    row.append(clean_val)
            if row:
                result.append(row)

        return result if result else text

    except Exception as e:
        return text  # Fallback

def normalize_result(value):
    """Normalizza il campo 'result' se contiene JSON o dict malformati."""
    if not isinstance(value, str):
        return value
    value = value.strip()
    if value.startswith('[['):
        return value  # già una lista
    if 'ordered_entries' in value:
        return extract_ordered_entries(value)
    return value

def process_file(csv_path):
    df = pd.read_csv(csv_path)

    # Aggiunge colonna 'model' se non esiste
    if 'model' not in df.columns:
        model_name = detect_model(os.path.basename(csv_path))
        df.insert(0, 'model', model_name)

    # Applica la trasformazione alla colonna 'result'
    if 'result' in df.columns:
        df['result'] = df['result'].apply(normalize_result)

    # 🔁 SALVA IL FILE
    df.to_csv(csv_path, index=False)
    print(f"✅ Salvato: {csv_path}")

def main():
    for file in os.listdir(CSV_DIR):
        if file.endswith('.csv'):
            full_path = os.path.join(CSV_DIR, file)
            process_file(full_path)

if __name__ == '__main__':
    main()