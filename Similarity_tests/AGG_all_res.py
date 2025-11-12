import os
import pandas as pd
from pathlib import Path

def extract_model_from_filename(filename):
    """
    Estrae il nome del modello dal nome del file.
    
    Args:
        filename: Nome del file (stringa)
    
    Returns:
        Nome del modello o None
    """
    filename_lower = filename.lower()
    
    if 'gemini' in filename_lower:
        if 'pro' in filename_lower:
            return 'Gemini 2.5 Pro'
        elif 'flash' in filename_lower:
            return 'Gemini 2.5 Flash'
        else:
            return 'Gemini'
    
    return None

def extract_ds_shape_from_path(file_path):
    """
    Estrae il valore di ds_shape dal percorso del file.
    
    Args:
        file_path: Percorso completo del file (Path object)
    
    Returns:
        Valore di ds_shape (stringa)
    """
    path_str = str(file_path)
    
    if 'cut_300' in path_str:
        return '30x10'
    elif 'cut_1000' in path_str:
        return '100x10'
    else:
        return '4000x10'

def get_removed_columns(folder_source):
    """
    Determina le colonne rimosse in base al folder_source.
    
    Args:
        folder_source: Nome della cartella sorgente
    
    Returns:
        Lista di colonne rimosse
    """
    mapping = {
        'no_title': ['title'],
        'no_all': ['year', 'genre', 'directors', 'writers', 'main_cast', 
                   'duration_min', 'AVG_score', 'number_of_votes'],
        'no_id': ['IMDB_id']
    }
    
    return mapping.get(folder_source, [])

def process_csv_files(base_folder, output_file='unified_results.csv'):
    """
    Processa tutti i file CSV in una cartella e sottocartelle, li unisce
    e aggiunge le colonne folder_source e rem_col.
    
    Args:
        base_folder: Percorso della cartella principale (es: '/risultati/all')
        output_file: Nome del file CSV di output
    """
    
    # Estrai il nome della cartella principale
    folder_name = Path(base_folder).name
    
    # Lista per contenere tutti i dataframe
    all_dfs = []
    
    # Trova tutti i file CSV ricorsivamente
    csv_files = list(Path(base_folder).rglob('*.csv'))
    
    if not csv_files:
        print(f"Nessun file CSV trovato in {base_folder}")
        return
    
    print(f"Trovati {len(csv_files)} file CSV")
    
    # Leggi e combina tutti i CSV
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file, keep_default_na=False, na_values=[''])
            
            # Rinomina execu_type in exec_type se presente
            if 'execu_type' in df.columns:
                df.rename(columns={'execu_type': 'exec_type'}, inplace=True)
                print(f"Rinominato 'execu_type' -> 'exec_type' in {csv_file.name}")
            
            # Converti exec_type in stringa e gestisci valori nulli
            if 'exec_type' in df.columns:
                df['exec_type'] = df['exec_type'].astype(str)
                # Sostituisci stringhe vuote con 'NULL' se necessario
                df['exec_type'] = df['exec_type'].replace('', 'NULL')
            
            # Se la colonna model non esiste, estrai dal nome del file
            if 'model' not in df.columns:
                model_name = extract_model_from_filename(csv_file.name)
                if model_name:
                    df['model'] = model_name
                    print(f"Aggiunta colonna 'model' = '{model_name}' per {csv_file.name}")
                else:
                    df['model'] = 'Unknown'
                    print(f"⚠️  Impossibile estrarre il modello da {csv_file.name}, impostato come 'Unknown'")
            
            # Se la colonna ds_shape non esiste, estrai dal percorso
            if 'ds_shape' not in df.columns:
                ds_shape = extract_ds_shape_from_path(csv_file)
                df['ds_shape'] = ds_shape
                print(f"Aggiunta colonna 'ds_shape' = '{ds_shape}' per {csv_file.name}")
            
            # Aggiungi la colonna con il nome della cartella principale
            df['folder_source'] = folder_name
            
            # Aggiungi la colonna rem_col con le colonne rimosse
            removed_cols = get_removed_columns(folder_name)
            df['rem_col'] = str(removed_cols)
            
            all_dfs.append(df)
            print(f"Letto: {csv_file}")
        except Exception as e:
            print(f"Errore leggendo {csv_file}: {e}")
    
    if not all_dfs:
        print("Nessun dataframe caricato con successo")
        return
    
    # Unisci tutti i dataframe
    unified_df = pd.concat(all_dfs, ignore_index=True)
    print(f"\nDataframe unificato: {len(unified_df)} righe")
    
    # Salva il risultato
    unified_df.to_csv(output_file, index=False)
    print(f"\nFile unificato salvato in: {output_file}")
    
    # Mostra i modelli trovati
    print(f"\nModelli presenti nel dataset:")
    print(unified_df['model'].unique())
    
    # Mostra i valori di exec_type trovati
    print(f"\nValori di exec_type presenti:")
    print(unified_df['exec_type'].unique())
    
    # Mostra i valori di ds_shape trovati
    print(f"\nValori di ds_shape presenti:")
    print(unified_df['ds_shape'].unique())
    
    # Mostra i valori di folder_source trovati
    print(f"\nValori di folder_source presenti:")
    print(unified_df['folder_source'].unique())
    
    # Mostra i valori di rem_col trovati
    print(f"\nValori di rem_col presenti:")
    print(unified_df['rem_col'].unique())
    
    return unified_df

# Esempio di utilizzo
if __name__ == "__main__":
    # Modifica questo percorso con la tua cartella
    base_folder = "results/no_id"
    
    # Oppure usa il percorso corrente
    # base_folder = "."
    
    # Esegui il processing
    result = process_csv_files(base_folder, 'results/no_id/unified_results.csv')
    
    if result is not None:
        print("\nPrime righe del risultato:")
        print(result.head())