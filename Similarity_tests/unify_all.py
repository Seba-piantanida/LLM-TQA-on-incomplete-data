import pandas as pd
from pathlib import Path

def unify_unified_results(base_folder, output_file='final_unified_results.csv'):
    """
    Cerca tutti i file 'unified_results.csv' in una cartella e sottocartelle
    e li unisce in un unico file.
    
    Args:
        base_folder: Percorso della cartella principale dove cercare
        output_file: Nome del file CSV di output
    """
    
    # Lista per contenere tutti i dataframe
    all_dfs = []
    
    # Trova tutti i file 'unified_results.csv' ricorsivamente
    unified_files = list(Path(base_folder).rglob('unified_results.csv'))
    
    if not unified_files:
        print(f"Nessun file 'unified_results.csv' trovato in {base_folder}")
        return
    
    print(f"Trovati {len(unified_files)} file 'unified_results.csv'\n")
    
    # Leggi e combina tutti i file
    for unified_file in unified_files:
        try:
            df = pd.read_csv(unified_file, keep_default_na=False, na_values=[''])
            all_dfs.append(df)
            print(f"Letto: {unified_file} ({len(df)} righe)")
        except Exception as e:
            print(f"❌ Errore leggendo {unified_file}: {e}")
    
    if not all_dfs:
        print("\n❌ Nessun dataframe caricato con successo")
        return
    
    # Unisci tutti i dataframe
    final_df = pd.concat(all_dfs, ignore_index=True)
    print(f"\n✅ Dataframe finale unificato: {len(final_df)} righe totali")
    
    # Salva il risultato
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(output_file, index=False)
    print(f"✅ File finale salvato in: {output_file}")
    
    # Statistiche
    print(f"\n📊 Statistiche del dataset finale:")
    
    if 'folder_source' in final_df.columns:
        print(f"\nDistribuzione per folder_source:")
        print(final_df['folder_source'].value_counts())
    
    if 'model' in final_df.columns:
        print(f"\nModelli presenti:")
        print(final_df['model'].value_counts())
    
    if 'ds_shape' in final_df.columns:
        print(f"\nDistribuzione per ds_shape:")
        print(final_df['ds_shape'].value_counts())
    
    if 'exec_type' in final_df.columns:
        print(f"\nDistribuzione per exec_type:")
        print(final_df['exec_type'].value_counts())
    
    print(f"\nColonne presenti nel dataset finale:")
    print(final_df.columns.tolist())
    
    return final_df

# Esempio di utilizzo
if __name__ == "__main__":
    # Modifica questo percorso con la tua cartella principale
    base_folder = "results"
    
    # Specifica dove salvare il file finale
    output_file = "results/final_unified_results.csv"
    
    # Esegui l'unificazione
    result = unify_unified_results(base_folder, output_file)
    
    if result is not None:
        print("\n📋 Prime 5 righe del risultato finale:")
        print(result.head())
        
        print("\n📋 Ultime 5 righe del risultato finale:")
        print(result.tail())