import pandas as pd
from tqdm import tqdm
from ollama import chat, ChatResponse
import sqlite3
from dotenv import load_dotenv
import json
from together import Together
from pydantic import BaseModel, Field
import re
from enum import Enum
import os



output_schema = """
  "table_name": "table name",
  "ordered_entries": [
    {
      entry0
    },
    {
      entry1
    },
    {
      entry2
    }
  ]
}"""

schema = {
    "type": "object",
    "properties": {
        "table_name": {
            "type": "string"
        },
        "ordered_entries": {
            "type": "array",
            "items": {
                "type": "string"
            }
        }
    },
    "required": [
        "table_name",
        "ordered_entries"
    ]
}

def extract_json(output): 
        json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', output, re.DOTALL)

        if json_match:
            json_str = json_match.group(1).strip()
        else:
            json_match = re.search(r'(\{.*\})', output, re.DOTALL)
            if json_match:
                json_str = json_match.group(1).strip()
            else:
                raise ValueError("No JSON structure found in the output.")

        try:
            parsed_json = json.loads(json_str)
            return parsed_json.get('ordered_entries', {})
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON parsing error: {e}")



class QueryExecutor():

    class ExecType(Enum):
        NORMAL = 'normal'
        REMOVE = 'remove'
        NULLABLE = 'nullable'


    def __init__(self, remote_models: list, tests: str, local_models : list = [], exec_type: ExecType = ExecType.NORMAL):
       
        self.models = remote_models
        self.tests = pd.read_csv(tests)
        self.local_models = local_models
        self.exec_type = exec_type
       

    def run(self)-> pd.DataFrame:

        
        result: pd.DataFrame = self.execute_API_queries()

        if self.local_models != []:
           local_result: pd.DataFrame = self.execute_local_query()
           result = pd.concat([result, local_result])
           

        sql_result: pd.DataFrame = self.execute_SQL_query()

        result = pd.merge(
            result,
            sql_result,
            how = 'left',
            on = 'query'
        )

        return result

    
    def execute_API_queries(self)-> pd.DataFrame:
            # vecchio prompt
            # system
            # "content": f"respond to the following queries using this data:\n {data},\n if some data are NULL or missing fillin the missing data. Only answer in JSON following the schema:\n {output_schema}",
            #            },
            # user
            # "content": f"{prompt}"

        #load API key
        load_dotenv()
        together = Together()

        answers = pd.DataFrame(columns=['db_path','table_name','test_category','query','model','AI_answer', 'SQL_query', 'failed_attempts', 'note'])
        prev_table = None
         
        for model in self.models:
            temp_answers = pd.DataFrame(columns=['db_path','table_name','test_category','query','model','AI_answer', 'SQL_query', 'failed_attempts', 'note'])
            for _, row in tqdm(self.tests.iterrows(), total=len(self.tests), colour="green", desc=f"API_queries on model: {model}"):
                
                if prev_table != row['tbl_name']:
                    data_set = pd.read_sql_table(row['tbl_name'], f'sqlite:///{row['db_path']}')
                data_set = self.get_dataset(row['tbl_name'], data_set, row['query'])
                prompt =  row['question'] 

                failed_attempts = 0
                

                for attempt in range(3):
                    try:
                        response = together.chat.completions.create(
                            temperature= 0.1,
                                messages=[
                                        {
                                            "role": "system",
                                            "content": "You are a highly skilled data analyst. Always follow instructions carefully. Always respond strictly in valid JSON."
                                        },
                                        {
                                            "role": "user",
                                            "content": f"""
                                        Objective:
                                        - Respond accurately to the provided query using the given dataset.
                                        - If any data fields are NULL, missing, or incomplete, **infer and fill** the missing information with the most logical and contextually appropriate value.
                                        - **Never leave fields empty or set to null.** Always provide the best inferred value based on the dataset context.

                                        Context:
                                        - Here is the dataset:\n{data_set}

                                        Query:
                                        - {prompt}

                                        Output Format:
                                        - Provide the response strictly in **valid JSON** format.
                                        - Follow exactly this schema:\n{output_schema}
                                        - Do not include any explanatory text or notes outside the JSON.
                                        - Ensure that all required fields are completed with non-null values.
                                        """
                                        }
                                    ],
                            model=model,
                            
                        )
                        AI_response = extract_json(response.choices[0].message.content)
                        temp_answers.loc[_] = [row['db_path'],row['tbl_name'],row['sql_tag'],prompt, model, [list(x.values()) for x in AI_response] , row['query'], failed_attempts, '']
                        break
                        
                        
                    except Exception as e:
                        failed_attempts += 1
                        if failed_attempts >=3:
                            try:
                                raw_output = response.choices[0].message.content
                            except:
                                raw_output = str(e) 

                            temp_answers.loc[_] = [row['db_path'],row['tbl_name'],row['sql_tag'],prompt, model, [], row['query'], failed_attempts, raw_output]
                            

                            break
                    
            answers = pd.concat([answers, temp_answers], ignore_index=True)

        return answers
    
    def execute_local_query(self)-> pd.DataFrame:
       
        answers = pd.DataFrame(columns=['db_path','table_name','test_category','query','model','AI_answer','SQL_query'])
        prev_table = None
        for model in self.local_models:
            for index, row in tqdm(self.tests.iterrows(), total=len(self.tests), colour="blue", desc=f"Local_queries on model: {model}"):
                prompt = row['question']
                if prev_table != row['tbl_name']:
                    data_set = pd.read_sql_table(row['tbl_name'], f'sqlite:///{row['db_path']}')
                data_set = self.get_dataset(row['tbl_name'], data_set, row['query'])
                
                response: ChatResponse = chat(model=model, 
                                    messages=[
                                        {
                                            "role": "system",
                                            "content": "You are a highly skilled data analyst. Always follow instructions carefully. Always respond strictly in valid JSON."
                                        },
                                        {
                                            "role": "user",
                                            "content": f"""
                                        Objective:
                                        - Respond accurately to the provided query using the given dataset.
                                        - If any data fields are NULL, missing, or incomplete, **infer and fill** the missing information with the most logical and contextually appropriate value.
                                        - **Never leave fields empty or set to null.** Always provide the best inferred value based on the dataset context.

                                        Context:
                                        - Here is the dataset:\n{data_set}

                                        Query:
                                        - {prompt}

                                        Output Format:
                                        - Provide the response strictly in **valid JSON** format.
                                        - Follow exactly this schema:\n{output_schema}
                                        - Do not include any explanatory text or notes outside the JSON.
                                        - Ensure that all required fields are completed with non-null values.
                                        """
                                        }
                                    ],
                                    format=schema)

                ordered_entries = json.loads(response['message']['content'])['ordered_entries']
                answers.loc[index] = [row['db_path'],row['tbl_name'],row['sql_tag'],query, model + '-local', ordered_entries, row['query']]
                
        return answers
    

    def execute_SQL_query(self)-> pd.DataFrame:

        answers = pd.DataFrame(columns=['query', 'SQL_answer']) 

        grouped = self.tests.groupby('db_path')

        for db_path, group_df in grouped:
            try:
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()

                for index, row in tqdm(group_df.iterrows(), total=len(group_df), colour="red", desc=f"SQL_queries on {db_path}"):
                    query = row['SQL_query']
                    try:
                        cursor.execute(query)
                        results = cursor.fetchall()
                        answers.loc[index] = [row['query'], results]
                    except sqlite3.Error as e:
                        print(f"Error executing query for row {index+1} in {db_path}: {e}")
            except sqlite3.Error as e:
                print(f"Could not connect to database {db_path}: {e}")
            finally:
                conn.close()

        return answers
    
    def get_dataset(self, table_name: str, data_set:pd.DataFrame, sql_query: str) -> pd.DataFrame:
        if self.exec_type == QueryExecutor.ExecType.NORMAL:
            return f"{table_name}: {data_set.to_json(index=False)}\n"
        
        elif self.exec_type == QueryExecutor.ExecType.REMOVE:

            match = re.search(r'ORDER BY\s+([^, \n`]+|`[^`]+`)', sql_query, re.IGNORECASE)
            if match:
                column = match.group(1)
                column = column.strip('`')

                data_set = data_set.drop(columns=[column], errors='ignore')

                
            return f"{table_name}:\n{data_set.to_json(index=False)}\n"

        else:
            match = re.search(r'ORDER BY\s+([^, \n`]+|`[^`]+`)', sql_query, re.IGNORECASE)
            if match:
                column = match.group(1)
                column = column.strip('`')
                data_set[column] = None
                

        return f"{table_name}: {data_set.to_json()}\n"
        




