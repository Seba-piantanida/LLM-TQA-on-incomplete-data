import pandas as pd

def clean_tests_from_results(tests_df: pd.DataFrame, results_df: pd.DataFrame) -> pd.DataFrame:

    res_filtered = results_df[results_df['execution_accuracy'] == 1].copy()
    model_set = set(results_df['model'].unique())

    valid_queries = [
        query for query, group in res_filtered.groupby('query')
        if set(group['model']) == model_set
    ]
    df_clean = tests_df[tests_df['question'].isin(valid_queries)].copy()
    
    return df_clean