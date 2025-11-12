import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from query_executor import QueryExecutor
import pandas as pd
from qatch.evaluate_dataset.orchestrator_evaluator import OrchestratorEvaluator

    
    
r_models = [ 
    'meta-llama/Llama-3.3-70B-Instruct-Turbo-Free',
    'deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free' ,
    ]

tests = 'variance_test/tests.csv'
exec_type = QueryExecutor.ExecType.NORMAL
exec = QueryExecutor(remote_models=r_models, tests=f'{tests}', dataBase='data/db_cut_300.sqlite', exec_type=exec_type)

res = exec.execute_API_queries()
res.to_json("temp_res.json")
res = OrchestratorEvaluator().evaluate_df(
        df=res,
        target_col_name='SQL_query',
        prediction_col_name='AI_answer',
        db_path_name='db_path',
    )
output_file = f"variance_test/variance_tests_QATCH.csv"
res.to_csv(output_file, index=False)
output_file = f"variance_tests_QATCH.csv"
res.to_csv(output_file, index=False)
