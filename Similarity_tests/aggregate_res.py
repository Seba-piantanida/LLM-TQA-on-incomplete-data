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

def process_csv_files(base_folder, output_file='aggregated_results.csv'):
    """
    Processa tutti i file CSV in una cartella e sottocartelle, li unisce,
    aggiunge il nome della cartella principale e aggrega i dati.
    
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
            all_dfs.append(df)
            print(f"Letto: {csv_file}")
        except Exception as e:
            print(f"Errore leggendo {csv_file}: {e}")
    
    if not all_dfs:
        print("Nessun dataframe caricato con successo")
        return
    
    # Unisci tutti i dataframe
    combined_df = pd.concat(all_dfs, ignore_index=True)
    print(f"\nDataframe combinato: {len(combined_df)} righe")
    
    # Identifica le colonne numeriche (escludendo le colonne di raggruppamento)
    numeric_columns = combined_df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    # Rimuovi eventuali colonne numeriche che non devono essere aggregate
    exclude_cols = ['test_id', 'retry_count']
    numeric_columns = [col for col in numeric_columns if col not in exclude_cols]
    
    print(f"Colonne numeriche per l'aggregazione: {numeric_columns}")
    
    # Aggrega per test_language, test_category, model, exec_type, ds_shape
    groupby_cols = ['test_language', 'test_category', 'model', 'exec_type', 'ds_shape', 'folder_source']
    
    # Crea il dizionario di aggregazione (media per tutte le colonne numeriche)
    agg_dict = {col: 'mean' for col in numeric_columns}
    
    # Aggiungi il conteggio delle righe
    agg_dict['test_id'] = 'count'
    
    # Esegui l'aggregazione
    aggregated_df = combined_df.groupby(groupby_cols, as_index=False).agg(agg_dict)
    
    # Rinomina la colonna test_id in num_tests
    aggregated_df.rename(columns={'test_id': 'num_tests'}, inplace=True)
    
    # Arrotonda i valori numerici a 4 decimali
    for col in numeric_columns:
        if col in aggregated_df.columns:
            aggregated_df[col] = aggregated_df[col].round(4)
    
    # Salva il risultato
    aggregated_df.to_csv(output_file, index=False)
    print(f"\nFile aggregato salvato in: {output_file}")
    print(f"Numero di righe aggregate: {len(aggregated_df)}")
    
    # Mostra i modelli trovati
    print(f"\nModelli presenti nel dataset:")
    print(aggregated_df['model'].unique())
    
    # Mostra i valori di exec_type trovati
    print(f"\nValori di exec_type presenti:")
    print(aggregated_df['exec_type'].unique())
    
    # Mostra i valori di ds_shape trovati
    print(f"\nValori di ds_shape presenti:")
    print(aggregated_df['ds_shape'].unique())
    
    return aggregated_df

# Esempio di utilizzo
if __name__ == "__main__":
    # Modifica questo percorso con la tua cartella
    base_folder = "results/no_title"
    
    # Oppure usa il percorso corrente
    # base_folder = "."
    
    # Esegui il processing
    result = process_csv_files(base_folder, 'results/no_title/aggregated_results.csv')
    
    if result is not None:
        print("\nPrime righe del risultato:")
        print(result.head())