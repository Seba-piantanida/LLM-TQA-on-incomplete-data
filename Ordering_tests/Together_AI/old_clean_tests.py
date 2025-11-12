import pandas as pd

def extract_common_valid_results(results: pd.DataFrame) -> pd.DataFrame:
   
    res_filtered = results[results['execution_accuracy'] == 1].copy()
    model_set = set(results['model'].unique())

    
    valid_queries = []
    for query, group in res_filtered.groupby('query'):
        if set(group['model']) == model_set:
            valid_queries.append(query)

    valid_results = res_filtered[res_filtered['query'].isin(valid_queries)].copy()

    return valid_results

def select_tests(df2, df_clean):
    valid_queries = set(df_clean['query'].unique())
    df_filtered = df2[df2['question'].isin(valid_queries)].copy()

    return df_filtered

tests = 'tests/test_300_comma.csv'
results = pd.read_csv('output/out_test_300_comma_normal_QATCH.csv')

valid_results = extract_common_valid_results(results)
print(f"{results.shape[0]} -> {valid_results.shape[0]}")

tests = pd.read_csv(tests)
results_clean = select_tests(tests, valid_results)
output_file = 'tests/test_300_comma_clean.csv'
results_clean.to_csv(output_file, index=False)
print(f"{tests.shape[0]} -> {results_clean.shape[0]}")