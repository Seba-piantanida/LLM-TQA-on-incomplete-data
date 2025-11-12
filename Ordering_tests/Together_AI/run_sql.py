import pandas as pd
import os
from query_executor import QueryExecutor  # Sostituisci 'your_module' con il nome effettivo del file .py dove sta la classe


# === CONFIGURAZIONE ===
INPUT_CSV = "output/categorical_simple/out_test_categorical_simple_normal_QATCH.csv"         # Percorso al file CSV con le domande
REMOTE_MODELS = []                     # Lascia vuoto se vuoi solo la parte SQL
LOCAL_MODELS = []                      # Idem, vuoto se non usi modelli locali


def main():
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"File CSV '{INPUT_CSV}' non trovato.")

    tests_df = pd.read_csv(INPUT_CSV)

    if 'db_path' not in tests_df.columns:
        raise ValueError("Il file CSV deve contenere una colonna 'db_path'.")

    all_results = pd.DataFrame()

    # Itera su ciascun database presente nel CSV
    for db_path in tests_df['db_path'].unique():
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"Database '{db_path}' non trovato.")

        # Filtra le righe per il database corrente
        filtered_csv_path = f"__filtered_{os.path.basename(db_path)}.csv"
        tests_df[tests_df['db_path'] == db_path].to_csv(filtered_csv_path, index=False)

        executor = QueryExecutor(
            remote_models=REMOTE_MODELS,
            tests=filtered_csv_path,
            local_models=LOCAL_MODELS,
            exec_type=QueryExecutor.ExecType.NORMAL
        )

        result = executor.execute_SQL_query()
        
        all_results = pd.concat([all_results, result], ignore_index=True)
        

        os.remove(filtered_csv_path)

    final_df = pd.merge(tests_df, all_results, on='query', how='left')
    
    final_df = final_df.drop_duplicates(subset=['query', 'model'])

    sql_query_index = final_df.columns.get_loc('SQL_query')

    # Rimuovi SQL_query
    final_df.drop(columns=['SQL_query'], inplace=True)

    # Inserisci SQL_answer in quella posizione e rinominala in SQL_query
    final_df.insert(sql_query_index, 'SQL_query',final_df['SQL_answer'])

    # (Opzionale) Rimuovi la colonna SQL_answer originale se non ti serve più
    final_df.drop(columns=['SQL_answer'], inplace=True)

   

    

    # Salva il file CSV finale
    base, ext = os.path.splitext(INPUT_CSV)
    output_csv = f"{base}_sql{ext}"
    final_df.to_csv(output_csv, index=False)

    print(f"CSV con risultati SQL salvato come: {output_csv}")


if __name__ == "__main__":
    main()