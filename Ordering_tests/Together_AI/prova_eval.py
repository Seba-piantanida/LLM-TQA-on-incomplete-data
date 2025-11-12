from query_executor import QueryExecutor
import pandas as pd
from qatch.evaluate_dataset.orchestrator_evaluator import OrchestratorEvaluator

res = pd.read_json('output/output_full_db_2mod.json', orient='records', lines=True)
def clean_list_of_dicts(nested_list):
    # Se è una lista
    if isinstance(nested_list, list):
        new_list = []
        for item in nested_list:
            if isinstance(item, list) and len(item) > 0 and isinstance(item[0], dict):
                # Prendo il valore del primo (e unico) dizionario
                value = list(item[0].values())[0]
                new_list.append([value])
            else:
                new_list.append(item)  # Se non è una lista di dizionari lascialo com'è
        return new_list
    else:
        return nested_list

# Applichiamo alla colonna AI_answer
res['AI_answer'] = res['AI_answer'].apply(clean_list_of_dicts)

res = OrchestratorEvaluator().evaluate_df(
        df=res,
        target_col_name='SQL_query',
        prediction_col_name='AI_answer',
        db_path_name='db_path',
    )
res.to_csv('output/output_full_db_2mod_QATCH.csv', index=False)
