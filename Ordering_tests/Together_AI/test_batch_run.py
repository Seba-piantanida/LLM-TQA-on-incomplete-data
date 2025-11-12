import json
import pandas as pd
from pathlib import Path
from query_executor import QueryExecutor
from qatch.evaluate_dataset.orchestrator_evaluator import OrchestratorEvaluator
from aggregate_res import compute_avg_res
from clean_tests import clean_tests_from_results
from tqdm import tqdm
import os
import requests
from dotenv import load_dotenv


def run_pipeline(test_path, out_path, models):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    exec_types = [
        QueryExecutor.ExecType.NORMAL,
        QueryExecutor.ExecType.NULLABLE,
        QueryExecutor.ExecType.REMOVE
    ]

    # === Step 1: Normal execution ===
    res = QueryExecutor(remote_models=models, tests=test_path,  exec_type=QueryExecutor.ExecType.NORMAL).execute_API_queries()
    res_file = Path(out_path) / f"out_{Path(test_path).stem}_{QueryExecutor.ExecType.NORMAL.value}_pre_val.csv"
    res.to_csv()
    res = OrchestratorEvaluator().evaluate_df(df=res, target_col_name='SQL_query', prediction_col_name='AI_answer', db_path_name='db_path')
    res_file = Path(out_path) / f"out_{Path(test_path).stem}_{QueryExecutor.ExecType.NORMAL.value}_QATCH.csv"
    res.to_csv(res_file, index=False)
    compute_avg_res(input_csv=res_file, output_csv=res_file.with_name(res_file.stem + "_summary.csv"))

    # === Step 2: Clean test file ===
    clean_test_file = Path(test_path).with_name(Path(test_path).stem + '_clean.csv')
    cleaned_df = clean_tests_from_results(tests_df=pd.read_csv(test_path), results_df=res)
    cleaned_df.to_csv(clean_test_file, index=False)

    # === Step 3: Nullable and Remove with cleaned tests ===
    for exec_type in [QueryExecutor.ExecType.NULLABLE, QueryExecutor.ExecType.REMOVE]:
        res = QueryExecutor(remote_models=models, tests=clean_test_file,  exec_type=exec_type).execute_API_queries()
        res_file = Path(out_path) / f"out_{Path(test_path).stem}_{QueryExecutor.ExecType.NORMAL.value}_pre_val.csv"
        res.to_csv()
        res = OrchestratorEvaluator().evaluate_df(df=res, target_col_name='SQL_query', prediction_col_name='AI_answer', db_path_name='db_path')
        res_file = Path(out_path) / f"out_{Path(test_path).stem}_clean_{exec_type.value}_QATCH.csv"
        res.to_csv(res_file, index=False)
        compute_avg_res(input_csv=res_file, output_csv=res_file.with_name(res_file.stem + "_summary.csv"))

    # === Step 4: Nullable with full tests ===
    res = QueryExecutor(remote_models=models, tests=test_path,  exec_type=QueryExecutor.ExecType.NULLABLE).execute_API_queries()
    res_file = Path(out_path) / f"out_{Path(test_path).stem}_{QueryExecutor.ExecType.NORMAL.value}_pre_val.csv"
    res.to_csv()
    res = OrchestratorEvaluator().evaluate_df(df=res, target_col_name='SQL_query', prediction_col_name='AI_answer', db_path_name='db_path')
    res_file = Path(out_path) / f"out_{Path(test_path).stem}_{QueryExecutor.ExecType.NULLABLE.value}_QATCH.csv"
    res.to_csv(res_file, index=False)
    compute_avg_res(input_csv=res_file, output_csv=res_file.with_name(res_file.stem + "_summary.csv"))


def main():
    with open('test_batch.json', 'r') as f:
        config = json.load(f)

    models = config.get("models", [])
    for test in tqdm(config.get("tests", []), desc=f"Running test cases", colour="red"):
        test_path = test["test_file"]
        out_path = test["out_path"]
        Path(out_path).mkdir(parents=True, exist_ok=True)
        run_pipeline(test_path, out_path, models)

if __name__ == "__main__":
    load_dotenv()
    
   
    main()
   