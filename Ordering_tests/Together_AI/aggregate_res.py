import pandas as pd


def compute_avg_res(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    df.to_csv(input_csv, index=False)

    group_cols = ['table_name', 'model']

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

if __name__ == "__main__":
    input_csv = 'output/out_test_simple_hand_normal_QATCH.csv'
    output_csv = f'{input_csv.replace(".csv", "_summary.csv")}'
    compute_avg_res(input_csv, output_csv)