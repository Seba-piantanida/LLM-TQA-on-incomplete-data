from query_executor import QueryExecutor
import pandas as pd
from qatch.evaluate_dataset.orchestrator_evaluator import OrchestratorEvaluator
from aggregate_res import compute_avg_res

    
    
r_models = [ 
    'meta-llama/Llama-3.3-70B-Instruct-Turbo-Free',
    'deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free' ,
    
    

    ]
# 'google/gemma-2-27b-it',
# 'meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8',
#     'Qwen/Qwen2-VL-72B-Instruct',
#     'scb10x/scb10x-llama3-1-typhoon2-70b-instruct'
tests = 'test_categorical_simple_cut_300.csv'
db_path = 'data/db_categorical_simple_cut_300.sqlite'
out_path = 'output/categorical_simple_cut'


exec_type = QueryExecutor.ExecType.REMOVE
exec = QueryExecutor(remote_models=r_models, tests=f'tests/{tests}', dataBase=db_path, exec_type=exec_type)

res = exec.execute_API_queries()
res.to_csv(out_path + 'last_out.csv')
res = OrchestratorEvaluator().evaluate_df(
        df=res,
        target_col_name='SQL_query',
        prediction_col_name='AI_answer',
        db_path_name='db_path',
    )
output_file = f"{out_path}/out_{tests.replace('.csv', '')}_{exec_type.value}_QATCH.csv"
res.to_csv(output_file, index=False)

compute_avg_res(input_csv=output_file, output_csv=f'{output_file.replace(".csv", "_summary.csv")}')