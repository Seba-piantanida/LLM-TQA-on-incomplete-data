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
import sqlparse
from sqlparse.sql import IdentifierList, Identifier
from sqlparse.tokens import Keyword, DML



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
        self.table_cache = {}  # Cache for loaded tables
       

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

    
    def get_first_order_column(self, sql_query: str) -> str | None:
        """
        Estrae in modo sicuro la prima colonna dall'ORDER BY.
        Gestisce virgole e concatenazioni con '+'. 
        Rimuove ASC/DESC, LIMIT, alias, qualificatori t.col, funzioni semplici,
        e tutti gli apici/backtick/brackets spaiati.
        """
        match = re.search(r'ORDER\s+BY\s+(.+?)(?:$|;)', sql_query, re.IGNORECASE)
        if not match:
            return None

        order_clause = match.group(1).strip()

        # Prendi solo la prima "espressione" separata da virgola
        first_expr = re.split(r',', order_clause)[0].strip()

        # Se c'è un '+', prendi solo la parte prima del '+'
        first_col_part = re.split(r'\+', first_expr)[0].strip()

        # Rimuovi parole chiave SQL
        first_col = re.sub(
            r'\s+(ASC|DESC|NULLS\s+FIRST|NULLS\s+LAST|LIMIT\s+\d+)\b',
            '',
            first_col_part,
            flags=re.IGNORECASE
        )

        # Togli tutti i possibili quote/apici/backtick/brackets all'inizio e alla fine
        first_col = first_col.strip('`"\'[] ')

        # Se è qualificato t.col -> prendi solo col
        if '.' in first_col:
            first_col = first_col.split('.')[-1].strip('`"\'[] ')

        # Se è una funzione semplice FUNC(col) -> prendi l'argomento
        func_match = re.match(r'^[A-Za-z_][\w_]*\(\s*([^)]+)\s*\)$', first_col)
        if func_match:
            first_col = func_match.group(1).strip('`"\'[] ')

        # Se è un numero (ORDER BY 1) -> ignora
        if first_col.isdigit():
            return None

        return first_col or None


    def extract_table_names(self,sql_query):
        """
        Extract table names from SQL query using sqlparse
        """
        try:
            parsed = sqlparse.parse( sql_query)[0]
            tables = set()
            
            def extract_from_token(token):
                if token.ttype is Keyword and token.value.upper() in ('FROM', 'JOIN', 'INTO', 'UPDATE'):
                    return True
                return False
            
            def extract_table_name(token):
                if isinstance(token, Identifier):
                    return token.get_name()
                elif token.ttype is None:
                    return token.value
                return None
            
            from_seen = False
            for token in parsed.flatten():
                if from_seen:
                    if token.ttype is Keyword:
                        from_seen = False
                    elif token.ttype is None and token.value.strip():
                        # Clean table name (remove quotes, aliases, etc.)
                        table_name = token.value.strip().split()[0]  # Take first part (before alias)
                        table_name = table_name.strip('`"\'[]')  # Remove quotes
                        if table_name and not table_name.upper() in ('AS', 'ON', 'WHERE', 'GROUP', 'ORDER', 'HAVING', 'LIMIT'):
                            tables.add(table_name)
                            from_seen = False
                elif extract_from_token(token):
                    from_seen = True
            
            # Fallback: simple regex extraction if sqlparse doesn't work well
            if not tables:
                # Extract after FROM, JOIN, INTO, UPDATE keywords
                from_pattern = r'(?:FROM|JOIN|INTO|UPDATE)\s+([`"\']?[\w_]+[`"\']?)'
                matches = re.findall(from_pattern, sql_query, re.IGNORECASE)
                for match in matches:
                    table_name = match.strip('`"\'[]').split()[0]
                    if table_name:
                        tables.add(table_name)
            
            return list(tables)
        
        except Exception as e:
            log(f"⚠️ Error parsing SQL query: {e}")
            # Fallback to original behavior
            return []



    def load_tables_for_query(self, db_path: str, sql_query: str, primary_table: str, exec_type: str, rem_col: list | None = None):
        """
        Carica tutte le tabelle necessarie alla query SQL.
        In base a exec_type:
        - REMOVE: rimuove la prima colonna dell'ORDER BY (se presente in una tabella)
        - NULL: setta a NULL tutti i valori della prima colonna dell'ORDER BY (se presente)
        - NORMAL: non modifica nulla
        """
        # Estrai i nomi delle tabelle dalla query
        table_names = self.extract_table_names(sql_query)
        if primary_table not in table_names:
            table_names.append(primary_table)
        if not table_names:
            table_names = [primary_table]

    

        # Trova la prima colonna nell'ORDER BY
        if rem_col is None or len(rem_col) == 0:
            first_order_column = self.get_first_order_column(sql_query)
            rem_col = [first_order_column] if first_order_column else []

        tables_data = {}

        for table_name in table_names:
            try:
                conn = sqlite3.connect(db_path)
                table_data = pd.read_sql_query(f"SELECT * FROM `{table_name}`", conn)
                conn.close()

                if first_order_column:
                    # confronto case-insensitive con i nomi reali delle colonne
                    cols_map = {c.lower(): c for c in table_data.columns}
                    key = first_order_column.lower()
                    if key in cols_map:
                        real_col = cols_map[key]
                        if exec_type == exec_type.REMOVE:
                            table_data = table_data.drop(columns=[real_col], errors='ignore')
                           
                        elif exec_type == exec_type.NULLABLE:
                            table_data[real_col] = None
                            
                
                tables_data[table_name] = table_data

            except Exception as e:
              
                continue

        return tables_data

    
    def format_dataset(self, tables_data: dict) -> str:
        """
        Format multiple tables data into a string for the prompt
        """
        formatted_data = ""
        for table_name, table_df in tables_data.items():
            print(table_name)
            
            
            formatted_data += f"{table_name}: {table_df.to_json()}\n\n"
        
        return formatted_data

    
    def execute_API_queries(self)-> pd.DataFrame:

        #load API key
        load_dotenv()
        API_KEY = os.getenv("TOGETHER_API_KEY")
        print(API_KEY)
        together = Together(api_key=API_KEY)

        answers = pd.DataFrame(columns=['db_path','table_name','test_category','query','model','AI_answer', 'SQL_query', 'failed_attempts', 'note'])
         
        for model in self.models:
            temp_answers = pd.DataFrame(columns=['db_path','table_name','test_category','query','model','AI_answer', 'SQL_query', 'failed_attempts', 'note'])
            for _, row in tqdm(self.tests.iterrows(), total=len(self.tests), colour="green", desc=f"API_queries on model: {model}"):

                rem_cols_list = row['rem_col'].split(',') if 'rem_col' in row and row['rem_col'] else None
                
                # Load all tables needed for this query
                tables_data = self.load_tables_for_query(row['db_path'], row['query'], row['tbl_name'], self.exec_type, rem_col= rem_cols_list)
                dataset_str = self.format_dataset(tables_data)

                prompt = row['question'] 
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
                                        - Respond accurately to the provided query using the given dataset(s).
                                        - If any data fields are NULL, missing, or incomplete, **infer and fill** the missing information with the most logical and contextually appropriate value.
                                        - **Never leave fields empty or set to null.** Always provide the best inferred value based on the dataset context.

                                        Context:
                                        - Here are the dataset(s):\n{dataset_str}

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
                        print('fail')
                        print(e)
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
        
        for model in self.local_models:
            for index, row in tqdm(self.tests.iterrows(), total=len(self.tests), colour="blue", desc=f"Local_queries on model: {model}"):
                prompt = row['question']
                
                # Load all tables needed for this query
                tables_data = self.load_tables_for_query(row['db_path'], row['query'], row['tbl_name'], self.exec_type)
                dataset_str = self.format_dataset(tables_data)
                
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
                                        - Respond accurately to the provided query using the given dataset(s).
                                        - If any data fields are NULL, missing, or incomplete, **infer and fill** the missing information with the most logical and contextually appropriate value.
                                        - **Never leave fields empty or set to null.** Always provide the best inferred value based on the dataset context.

                                        Context:
                                        - Here are the dataset(s):\n{dataset_str}

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
                answers.loc[index] = [row['db_path'],row['tbl_name'],row['sql_tag'],row['query'], model + '-local', ordered_entries, row['query']]
                
        return answers
    

    def execute_SQL_query(self)-> pd.DataFrame:

        answers = pd.DataFrame(columns=['query', 'SQL_answer']) 

        grouped = self.tests.groupby('db_path')

        for db_path, group_df in grouped:
            try:
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()

                for index, row in tqdm(group_df.iterrows(), total=len(group_df), colour="red", desc=f"SQL_queries on {db_path}"):
                    query = row['query']
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
        """
        This method is now deprecated in favor of load_tables_for_query and format_dataset
        Kept for backward compatibility
        """
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