from query_executor import QueryExecutor
import pandas as pd
from qatch.evaluate_dataset.orchestrator_evaluator import OrchestratorEvaluator
from aggregate_res import compute_avg_res
from clean_tests import *
from pathlib import Path
    
    
r_models = [ 
    'deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free',
    'meta-llama/Llama-3.3-70B-Instruct-Turbo-Free',
    ]
    
# 'deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free', 
# 'google/gemma-2-27b-it',
# 'meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8',
# 'Qwen/Qwen2-VL-72B-Instruct',
# 'scb10x/scb10x-llama3-1-typhoon2-70b-instruct'

tests = 'tests/test_numerical_simple.csv'
db_path = 'data/db_numerical_simple.sqlite'
out_path = "output/numerical_simple"

exec_type = QueryExecutor.ExecType.NORMAL
res = QueryExecutor(remote_models=r_models, tests=f'{tests}', dataBase=db_path, exec_type=exec_type).execute_API_queries()
#res.to_csv(f"{out_path}/pre_val/{tests.replace('.csv', '')}_{exec_type.value}.csv")
res = OrchestratorEvaluator().evaluate_df(
          df=res,
          target_col_name='SQL_query',
          prediction_col_name='AI_answer',
          db_path_name='db_path',
      )
res_file = f"{out_path}/out_{Path(tests).stem}_{exec_type.value}_QATCH.csv"
res.to_csv(res_file, index=False)
compute_avg_res(input_csv=res_file, output_csv=f'{res_file.replace(".csv", "_summary.csv")}')
res = pd.read_csv(res_file)
clean_test_file = f'{tests.replace('.csv', '_clean.csv')}'
clean_tests_from_results(tests_df=pd.read_csv(f'{tests}'), results_df=res).to_csv(f'{clean_test_file}')


null_res = QueryExecutor(remote_models=r_models, tests=f'{clean_test_file}', dataBase=db_path, exec_type=QueryExecutor.ExecType.NULLABLE).execute_API_queries()

null_res = OrchestratorEvaluator().evaluate_df(
        df=null_res,
        target_col_name='SQL_query',
        prediction_col_name='AI_answer',
        db_path_name='db_path',
    )
res_file = f"{out_path}/out_{Path(tests).stem}_clean_{QueryExecutor.ExecType.NULLABLE.value}_QATCH.csv"
null_res.to_csv(res_file, index=False)
compute_avg_res(input_csv=res_file, output_csv=f'{res_file.replace(".csv", "_summary.csv")}')



rem_res = QueryExecutor(remote_models=r_models, tests=f'{clean_test_file}', dataBase=db_path, exec_type=QueryExecutor.ExecType.REMOVE).execute_API_queries()

rem_res = OrchestratorEvaluator().evaluate_df(
        df=rem_res,
        target_col_name='SQL_query',
        prediction_col_name='AI_answer',
        db_path_name='db_path',
    )
res_file = f"{out_path}/out_{Path(tests).stem}_clean_{QueryExecutor.ExecType.REMOVE.value}_QATCH.csv"
rem_res.to_csv(res_file, index=False)
compute_avg_res(input_csv=res_file, output_csv=f'{res_file.replace(".csv", "_summary.csv")}')




null_full_res = QueryExecutor(remote_models=r_models, tests=f'{tests}', dataBase=db_path, exec_type=QueryExecutor.ExecType.NULLABLE).execute_API_queries()

null_full_res = OrchestratorEvaluator().evaluate_df(
        df=null_full_res,
        target_col_name='SQL_query',
        prediction_col_name='AI_answer',
        db_path_name='db_path',
    )
res_file = f"{out_path}/out_{Path(tests).stem}_{QueryExecutor.ExecType.NULLABLE.value}_QATCH.csv"
null_full_res.to_csv(res_file, index=False)
compute_avg_res(input_csv=res_file, output_csv=f'{res_file.replace(".csv", "_summary.csv")}')