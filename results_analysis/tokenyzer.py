import pandas as pd
import sqlite3
import json
import tiktoken
from typing import Dict, List, Any
import os

def get_table_schema_and_data(db_path: str, table_names: List[str]) -> Dict[str, Any]:
    """
    Estrae schema e dati dalle tabelle SQLite specificate.
    
    Args:
        db_path: Path al database SQLite
        table_names: Lista dei nomi delle tabelle da estrarre
    
    Returns:
        Dizionario con le informazioni delle tabelle in formato JSON
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        dataset = {}
        
        for table_name in table_names:
            table_name = table_name.strip()
            
            # Ottieni schema della tabella
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns_info = cursor.fetchall()
            
            if not columns_info:
                print(f"Warning: Table {table_name} not found in {db_path}")
                continue
            
            # Estrai nomi delle colonne
            columns = [col[1] for col in columns_info]
            
            # Ottieni tutti i dati della tabella
            cursor.execute(f"SELECT * FROM {table_name}")
            rows = cursor.fetchall()
            
            # Crea struttura dati per la tabella
            table_data = {
                "columns": columns,
                "rows": rows,
                "schema": {
                    col[1]: {
                        "type": col[2],
                        "not_null": bool(col[3]),
                        "default_value": col[4],
                        "primary_key": bool(col[5])
                    }
                    for col in columns_info
                }
            }
            
            dataset[table_name] = table_data
        
        conn.close()
        return dataset
        
    except sqlite3.Error as e:
        print(f"Error accessing database {db_path}: {e}")
        return {}
    except Exception as e:
        print(f"Unexpected error: {e}")
        return {}

def create_prompt(query: str, dataset: Dict[str, Any]) -> str:
    """
    Crea il prompt completo utilizzando il template specificato.
    
    Args:
        query: La query SQL da eseguire
        dataset: I dati delle tabelle in formato JSON
    
    Returns:
        Il prompt completo formattato
    """
    # Converti il dataset in stringa JSON formattata
    dataset_str = json.dumps(dataset, indent=2, ensure_ascii=False)
    
    # Schema di output
    output_schema = """
{
  "table_name": "table name",
  "ordered_entries": [
    {
      entry0
    },
    {
      entry1
    },
    {
      entry2
    }
  ]
}"""
    
    # Template del prompt
    prompt_template = """You are a highly skilled data analyst. Always follow instructions carefully. Always respond strictly in valid JSON.

Objective:
- Respond accurately to the provided query using the given dataset(s).
- If any data fields are NULL, missing, or incomplete, **infer and fill** the missing information with the most logical and contextually appropriate value.
- **Never leave fields empty or set to null.** Always provide the best inferred value based on the dataset context.

Context:
- Here are the dataset(s):
{dataset_str}

Query:
- {prompt}

Output Format:
- Provide the response strictly in **valid JSON** format.
- Follow exactly this schema:
{output_schema}
- Do not include any explanatory text or notes outside the JSON.
- Ensure that all required fields are completed with non-null values."""
    
    return prompt_template.format(
        dataset_str=dataset_str,
        prompt=query,
        output_schema=output_schema
    )

def count_tokens(text: str, model: str = "gpt-4") -> int:
    """
    Conta i token utilizzando tiktoken per il modello specificato.
    
    Args:
        text: Il testo di cui contare i token
        model: Il modello per cui contare i token (default: gpt-4)
    
    Returns:
        Numero di token
    """
    try:
        # Mappa dei modelli agli encoding
        model_encoding_map = {
            "gpt-4": "cl100k_base",
            "gpt-4-turbo": "cl100k_base", 
            "gpt-3.5-turbo": "cl100k_base",
            "text-davinci-003": "p50k_base",
            "text-davinci-002": "p50k_base",
            "code-davinci-002": "p50k_base"
        }
        
        # Usa l'encoding appropriato per il modello, default cl100k_base per GPT-4
        encoding_name = model_encoding_map.get(model, "cl100k_base")
        encoding = tiktoken.get_encoding(encoding_name)
        
        # Conta i token
        tokens = encoding.encode(text)
        return len(tokens)
        
    except Exception as e:
        print(f"Error counting tokens: {e}")
        # Fallback: stima approssimativa (1 token ≈ 4 caratteri per l'inglese)
        return len(text) // 4

def parse_table_names(table_name_str: str) -> List[str]:
    """
    Parse la stringa dei nomi delle tabelle che potrebbero essere separati da virgole.
    
    Args:
        table_name_str: Stringa contenente i nomi delle tabelle
    
    Returns:
        Lista dei nomi delle tabelle
    """
    if pd.isna(table_name_str):
        return []
    
    # Split per virgola e rimuovi spazi
    table_names = [name.strip() for name in str(table_name_str).split(',')]
    return [name for name in table_names if name]  # Rimuovi stringhe vuote

def process_csv_add_token_column(input_csv_path: str, output_csv_path: str = None):
    """
    Processa il CSV aggiungendo la colonna token_prompt_count.
    
    Args:
        input_csv_path: Path del file CSV di input
        output_csv_path: Path del file CSV di output (se None, sovrascrive l'input)
    """
    if output_csv_path is None:
        output_csv_path = input_csv_path
    
    # Leggi il CSV
    print(f"Leggendo il CSV: {input_csv_path}")
    df = pd.read_csv(input_csv_path)
    
    # Inizializza la nuova colonna
    token_counts = []
    
    # Processa ogni riga
    for index, row in df.iterrows():
        print(f"Processando riga {index + 1}/{len(df)}")
        
        try:
            db_path = row['db_path']
            table_name_str = row['table_name']
            query = row['query']
            model = row.get('model', 'gpt-4')  # Default a gpt-4 se non specificato
            
            # Verifica che i campi richiesti esistano
            if pd.isna(db_path) or pd.isna(table_name_str) or pd.isna(query):
                print(f"Riga {index}: Campi mancanti, usando 0 token")
                token_counts.append(0)
                continue
            
            # Verifica che il database esista
            if not os.path.exists(db_path):
                print(f"Riga {index}: Database {db_path} non trovato, usando 0 token")
                token_counts.append(0)
                continue
            
            # Parse i nomi delle tabelle
            table_names = parse_table_names(table_name_str)
            if not table_names:
                print(f"Riga {index}: Nessuna tabella valida trovata, usando 0 token")
                token_counts.append(0)
                continue
            
            # Estrai i dati delle tabelle
            dataset = get_table_schema_and_data(db_path, table_names)
            if not dataset:
                print(f"Riga {index}: Impossibile estrarre dati dalle tabelle, usando 0 token")
                token_counts.append(0)
                continue
            
            # Crea il prompt completo
            full_prompt = create_prompt(query, dataset)
            
            # Conta i token
            token_count = count_tokens(full_prompt, model)
            token_counts.append(token_count)
            
            print(f"Riga {index}: {token_count} token calcolati")
            
        except Exception as e:
            print(f"Errore nella riga {index}: {e}")
            token_counts.append(0)
    
    # Aggiungi la nuova colonna al DataFrame
    df['token_prompt_count'] = token_counts
    
    # Salva il nuovo CSV
    print(f"Salvando il CSV modificato: {output_csv_path}")
    df.to_csv(output_csv_path, index=False)
    
    print(f"Processo completato! Statistiche:")
    print(f"- Righe processate: {len(df)}")
    print(f"- Token totali calcolati: {sum(token_counts)}")
    print(f"- Token medi per prompt: {sum(token_counts) / len(token_counts):.2f}")
    print(f"- Token minimi: {min(token_counts)}")
    print(f"- Token massimi: {max(token_counts)}")

def main():
    """
    Funzione principale per eseguire lo script.
    """
    print("=== Token Count Calculator per Dataset SQL ===")
    print()
    
    # Percorso del file CSV di input
    input_csv = input("Inserisci il path del file CSV di input: ").strip()
    
    if not os.path.exists(input_csv):
        print(f"Errore: File {input_csv} non trovato!")
        return
    
    # Percorso del file CSV di output (opzionale)
    output_csv = input("Inserisci il path del file CSV di output (Enter per sovrascrivere): ").strip()
    if not output_csv:
        output_csv = None
    
    try:
        # Processa il CSV
        process_csv_add_token_column(input_csv, output_csv)
        
    except Exception as e:
        print(f"Errore durante il processo: {e}")

if __name__ == "__main__":
    # Verifica che tiktoken sia installato
    try:
        import tiktoken
    except ImportError:
        print("Errore: tiktoken non è installato. Installalo con: pip install tiktoken")
        exit(1)
    
    main()