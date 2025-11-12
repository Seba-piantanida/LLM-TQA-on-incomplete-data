import pandas as pd

df = pd.read_csv("results_v4/all_combined.csv")

df = df.groupby(['test_category', 'model', 'execution_type']).agg({
    'cell_precision': 'mean',
    'cell_recall': 'mean',
    'execution_accuracy': 'mean',
    'tuple_cardinality': 'mean',
    'tuple_constraint': 'mean',
    'tuple_order': 'mean'})

df.to_csv("results_v3/all_combined_aggregated.csv")