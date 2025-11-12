import pandas as pd
import os
from pathlib import Path

def trova_csv_validi(cartella_principale):
    """
    Trova tutti i file CSV nelle cartelle e sottocartelle,
    escludendo quelli con 'avg' nel nome.
    """
    csv_files = []
    for root, dirs, files in os.walk(cartella_principale):
        for file in files:
            if file.endswith('.csv') and 'avg' not in file.lower():
                csv_files.append(os.path.join(root, file))
    return csv_files

def leggi_e_filtra_csv(file_path, colonne_richieste):
    """
    Legge un CSV e verifica se ha le colonne richieste.
    Ritorna il dataframe con le colonne nell'ordine corretto, o None se non valido.
    """
    try:
        # Legge il CSV mantenendo 'NULL' come stringa
        df = pd.read_csv(file_path, keep_default_na=False, na_values=[''])
        
        # Verifica se tutte le colonne richieste sono presenti
        if all(col in df.columns for col in colonne_richieste):
            # Riorganizza le colonne nell'ordine specificato
            df_ordinato = df[colonne_richieste].copy()
            
            # Assicura che 'NULL' nella colonna execution_type rimanga come stringa
            if 'execution_type' in df_ordinato.columns:
                df_ordinato['execution_type'] = df_ordinato['execution_type'].astype(str)
            
            return df_ordinato
        else:
            print(f"File saltato (colonne mancanti): {file_path}")
            return None
    except Exception as e:
        print(f"Errore nella lettura di {file_path}: {e}")
        return None

def main():
    # Colonne richieste nell'ordine desiderato
    colonne_richieste = [
        'model', 'execution_type', 'table', 'sql_query', 'question',
        'result', 'valid_efficiency_score', 'cell_precision', 'cell_recall',
        'execution_accuracy', 'tuple_cardinality', 'tuple_constraint', 'tuple_order'
    ]
    
     # Percorsi - MODIFICA QUESTI PERCORSI
    cartella_csv = 'results_UDF/UDF'  # Cartella principale con i CSV
    file_join = 'tests/UDF/all_UDF_tests.csv'  # File con test_category
    file_output = 'results_UDF/UDF/UDF_all_res.csv'  # File di output
    
    print("Cerco i file CSV...")
    csv_files = trova_csv_validi(cartella_csv)
    print(f"Trovati {len(csv_files)} file CSV (escludendo quelli con 'avg' nel nome)")
    
    # Lista per raccogliere i dataframe validi
    dataframes = []
    
    print("\nLeggo e filtro i file...")
    for file_path in csv_files:
        df = leggi_e_filtra_csv(file_path, colonne_richieste)
        if df is not None:
            dataframes.append(df)
            print(f"✓ Aggiunto: {file_path}")
    
    if not dataframes:
        print("\nNessun file valido trovato!")
        return
    
    # Unisce tutti i dataframe
    print(f"\nUnisco {len(dataframes)} file...")
    df_unificato = pd.concat(dataframes, ignore_index=True)
    print(f"Dataframe unificato: {len(df_unificato)} righe")
    
    # Join con il file delle categorie
    print(f"\nEseguo il join con {file_join}...")
    try:
        df_categorie = pd.read_csv(file_join, keep_default_na=False, na_values=[''])
        
        if 'test_category' not in df_categorie.columns:
            print("ATTENZIONE: La colonna 'test_category' non è presente nel file di join!")
            return
        
        if 'question' not in df_categorie.columns:
            print("ATTENZIONE: La colonna 'question' non è presente nel file di join!")
            return
        
        # Join mantenendo solo la colonna test_category
        df_finale = df_unificato.merge(
            df_categorie[['question', 'test_category']], 
            on='question', 
            how='left'
        )
        
        print(f"Join completato: {len(df_finale)} righe")
        
        # Salva il risultato mantenendo 'NULL' come stringa
        df_finale.to_csv(file_output, index=False)
        print(f"\n✓ File salvato con successo: {file_output}")
        print(f"  Righe totali: {len(df_finale)}")
        print(f"  Colonne: {len(df_finale.columns)}")
        
    except FileNotFoundError:
        print(f"ERRORE: File non trovato: {file_join}")
    except Exception as e:
        print(f"ERRORE durante il join: {e}")

if __name__ == "__main__":
    main()