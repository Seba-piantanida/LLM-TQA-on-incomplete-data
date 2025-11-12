import json
import pandas as pd
from pathlib import Path
from query_executor import QueryExecutor
from tqdm import tqdm
import os
import requests
from dotenv import load_dotenv

test_path = 'test_UDF_all.json'


def run_pipeline(test_path, out_path, models, exec_t):
    # Crea la cartella output_pre_val_UDF invece di out_path
    output_dir = Path("output_pre_val_UDF")
    output_dir.mkdir(parents=True, exist_ok=True)

    exec_types = [
        QueryExecutor.ExecType.NORMAL,
        QueryExecutor.ExecType.NULLABLE,
        QueryExecutor.ExecType.REMOVE
    ]
    if exec_t.lower() == 'normal':
        print('performing normal tests')
        # === Step 1: Normal execution ===
        res = QueryExecutor(remote_models=models, tests=test_path, exec_type=QueryExecutor.ExecType.NORMAL).execute_API_queries()
        res_file = output_dir / f"out_{Path(test_path).stem}_{QueryExecutor.ExecType.NORMAL.value}_pre_val.csv"
        res.to_csv(res_file, index=False)

    # === Step 2: Nullable execution ===
    elif exec_t.lower() == 'nullable':
        print('performing null tests')
        res = QueryExecutor(remote_models=models, tests=test_path, exec_type=QueryExecutor.ExecType.NULLABLE).execute_API_queries()
        res_file = output_dir / f"out_{Path(test_path).stem}_{QueryExecutor.ExecType.NULLABLE.value}_pre_val.csv"
        res.to_csv(res_file, index=False)

    # === Step 3: Remove execution ===
    elif exec_t.lower() == 'remove':
        print('performing remove tests')
        res = QueryExecutor(remote_models=models, tests=test_path, exec_type=QueryExecutor.ExecType.REMOVE).execute_API_queries()
        res_file = output_dir / f"out_{Path(test_path).stem}_{QueryExecutor.ExecType.REMOVE.value}_pre_val.csv"
        res.to_csv(res_file, index=False)
    
    else:
        print("invalid exec type")


def main():
    global test_path
    with open(test_path, 'r') as f:
        config = json.load(f)

    models = config.get("models", [])
    for test in tqdm(config.get("tests", []), desc=f"Running test cases", colour="red"):
        test_path = test["test_file"]
        out_path = test["out_path"] 
        exec_type =  test["test_types"] # Non più utilizzato, ma mantenuto per compatibilità
        for exec in exec_type:
            run_pipeline(test_path, out_path, models, exec)


if __name__ == "__main__":
    load_dotenv()

    
    main()
   