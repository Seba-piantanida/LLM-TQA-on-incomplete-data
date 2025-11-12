import json
import pandas as pd
from pathlib import Path
from query_executor import QueryExecutor
from qatch.evaluate_dataset.orchestrator_evaluator import OrchestratorEvaluator
from tqdm import tqdm
import os

# === CONFIGURAZIONE ===
n_runs = 10
metrics_to_track = [
    "cell_precision", "cell_recall", "execution_accuracy",
    "tuple_cardinality", "tuple_constraint", "tuple_order"
]


def run_single_evaluation(models, test_path, exec_type):
    res = QueryExecutor(remote_models=models, tests=test_path, exec_type=exec_type).execute_API_queries()
    res_eval = OrchestratorEvaluator().evaluate_df(
        df=res,
        target_col_name='SQL_query',
        prediction_col_name='AI_answer',
        db_path_name='db_path'
    )
    return res_eval[metrics_to_track].mean().to_dict()


def run_pipeline(test_path, out_path, models, label):
    out_path = Path(out_path)
    out_path.mkdir(parents=True, exist_ok=True)
    test_name = Path(test_path).stem

    exec_types = [
        ("NORMAL", QueryExecutor.ExecType.NORMAL),
        ("NULLABLE", QueryExecutor.ExecType.NULLABLE),
        
    ]
    #("REMOVE", QueryExecutor.ExecType.REMOVE)

    for exec_type_name, exec_type_enum in exec_types:
        run_label = f"{exec_type_name}_{label}"
        metrics_list = []

        intermediate_file = out_path / f"intermediate_results_{run_label}_{test_name}.csv"

        for run_id in range(n_runs):
            metrics = run_single_evaluation(models, test_path, exec_type_enum)
            metrics["run"] = run_id
            metrics["exec_type"] = run_label
            metrics_list.append(metrics)

            # Salvataggio intermedio
            pd.DataFrame([metrics]).to_csv(intermediate_file, mode='a', index=False, header=not intermediate_file.exists())

        # Calcolo varianza
        df_metrics = pd.DataFrame(metrics_list)
        variances = df_metrics[metrics_to_track].var().to_dict()
        variances["exec_type"] = run_label

        # Salvataggio riepilogo varianza
        summary_file = out_path / f"variance_summary_{label}_{test_name}.csv"
        if summary_file.exists():
            summary_df = pd.read_csv(summary_file)
            summary_df = pd.concat([summary_df, pd.DataFrame([variances])], ignore_index=True)
        else:
            summary_df = pd.DataFrame([variances])

        summary_df = summary_df[["exec_type"] + metrics_to_track]
        summary_df.to_csv(summary_file, index=False)


def main():
    with open('test_batch_variance.json', 'r') as f:
        config = json.load(f)

    models = config.get("models", [])
    for test in tqdm(config.get("tests", []), desc="Running test cases", colour="red"):
        test_path = test["test_file"]
        out_path = test["out_path"]
        label = test["label"]
        run_pipeline(test_path, out_path, models, label)


if __name__ == "__main__":
    main()