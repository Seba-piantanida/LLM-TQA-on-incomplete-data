import os
import pandas as pd
import ast

from qatch.evaluate_dataset.orchestrator_evaluator import OrchestratorEvaluator

def convert_result_column(result):
    """Converte stringhe di liste in vere liste, se necessario."""
    if isinstance(result, list):
        return result
    try:
        return ast.literal_eval(result)
    except Exception:
        return result

def compute_avg_res(input_csv, output_csv):
    """Calcola medie raggruppate su colonne specifiche."""
    df = pd.read_csv(input_csv)
    df.to_csv(input_csv, index=False)

    group_cols = ['table', 'model']
    avg_cols = [
        'valid_efficiency_score',
        'cell_precision',
        'cell_recall',
        'execution_accuracy',
        'tuple_cardinality',
        'tuple_constraint',
        'tuple_order'
    ]

    summary_df = df.groupby(group_cols)[avg_cols].mean().reset_index()
    summary_df.rename(columns={col: f'avg_{col}' for col in avg_cols}, inplace=True)
    summary_df.to_csv(output_csv, index=False)
    print(f'Average results saved to {output_csv}')

def process_file(out_path):
    """
    Valuta un file CSV con OrchestratorEvaluator e calcola anche la media.
    Restituisce i percorsi dei file generati.
    """
    print(f'Processing {out_path}...')
    out_file = pd.read_csv(out_path)

    # Converti result in lista (se necessario)
    out_file['result'] = out_file['result'].apply(convert_result_column)

    # Esegui valutazione
    res = OrchestratorEvaluator().evaluate_df(
        df=out_file,
        target_col_name='sql_query',
        prediction_col_name='result',
        db_path_name='db_path',
    )

    # Salva risultati dettagliati
    res_path = out_path.replace('output', 'results')
    os.makedirs(os.path.dirname(res_path), exist_ok=True)
    res.to_csv(res_path, index=False)
    print(f'Saved detailed results to {res_path}')

    # Calcola medie
    avg_path = res_path.replace('.csv', '_avg.csv')
    compute_avg_res(res_path, avg_path)

    return res_path, avg_path

def process_all_files(input_dir):
    """Scansiona ricorsivamente tutti i CSV e li elabora uno per uno."""
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.csv'):
                out_path = os.path.join(root, file)
                try:
                    process_file(out_path)
                except Exception as e:
                    print(f'Error processing {out_path}: {e}')

if __name__ == '__main__':
    input_dir = 'output/tuple_limit/clean'  # Cambia questa directory secondo necessità
    process_all_files(input_dir)