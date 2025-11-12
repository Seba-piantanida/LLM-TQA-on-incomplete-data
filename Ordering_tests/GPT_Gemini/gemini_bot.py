import pandas as pd
import sqlite3
import re
import json
import os
import time
import random
import google.generativeai as genai
import compute_res
import requests

# === CONFIG ===
test_path = 'tests/test_numerical_simple.csv'
out_path = 'output/type_fix/categorical_simple/categorical_type_fix_clean_null_gemini-2.5_flash.csv'
EXEC_TYPE = 'NULL'  # 'REMOVE', 'NULL', 'NORMAL'
progress_path = 'progress_gemini.json'
GEMINI_MODEL = "gemini-2.5-flash"
API_KEY = os.getenv("GEMINI_API_KEY")
attesa = 6

# === INIT GOOGLE GEMINI ===

model = genai.GenerativeModel(GEMINI_MODEL)

# === GESTIONE PROGRESSO ===
if os.path.exists(progress_path):
    with open(progress_path, 'r') as f:
        progress = json.load(f)
else:
    progress = {}

last_completed_idx = int(progress.get(test_path, -1))

df = pd.read_csv(test_path, dtype=str).dropna(subset=['query', 'question'])
df = df[df.index > last_completed_idx]

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

def extract_json(output: str):
    output = output.strip()
    code_block_match = re.search(r'```(?:json)?\s*([\s\S]+?)\s*```', output)
    if code_block_match:
        json_str = code_block_match.group(1)
    else:
        json_match = re.search(r'(\{[\s\S]+?\})', output)
        json_str = json_match.group(1) if json_match else output
    json_str = re.sub(r'^(json|jsonCopiaModifica)?\s*', '', json_str).strip()
    try:
        parsed_json = json.loads(json_str)
        if "ordered_entries" not in parsed_json:
            raise ValueError("'ordered_entries' missing")
        return parsed_json
    except json.JSONDecodeError as e:
        raise ValueError(f"parsing error:\n{e}\n\nContenuto:\n{json_str[:500]}")

def get_dataset(table_name: str, db_path: str, sql_query: str) -> str:
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(f"SELECT * FROM `{table_name}`", conn)
    conn.close()
    match = re.search(r'ORDER BY\s+([^, \n`]+|`[^`]+`)', sql_query, re.IGNORECASE)
    if match:
        column = match.group(1).strip('`')
        if EXEC_TYPE == 'REMOVE':
            df = df.drop(columns=[column], errors='ignore')
        elif EXEC_TYPE == 'NULL' and column in df.columns:
            df[column] = None
    return f"{table_name}: {df.to_json(index=False)}\n"

# === CICLO PRINCIPALE ===
for idx, row in df.iterrows():
    while True:  # ciclo per retry in caso di errore
        try:
            db_path = row['db_path']
            table = row['tbl_name']
            sql = row['query']
            prompt = row['question']
            dataset_json = get_dataset(table, db_path, sql)

            full_prompt = f"""
You are a highly skilled data analyst. Always follow instructions carefully. Always respond strictly in valid JSON.

Objective:
- Respond accurately to the provided query using the given dataset.
- If any data fields are NULL, missing, or incomplete, **infer and fill** the missing information with the most logical and contextually appropriate value.
- **Never leave fields empty or set to null.** Always provide the best inferred value based on the dataset context.

Context:
- Here is the dataset:\n{dataset_json}

Query:
- {prompt}

Output Format:
- Provide the response strictly in **valid JSON** format.
- Follow exactly this schema:\n{output_schema}
- Do not include any explanatory text or notes outside the JSON.
- Ensure that all required fields are completed with non-null values.
"""

            response = model.generate_content(full_prompt)
            output = response.text

            try:
                parsed = extract_json(output)
                entries = parsed["ordered_entries"]
                array_of_arrays = [[entry[key] for key in entry] for entry in entries]
                result_to_save = array_of_arrays
            except Exception as e:
                print(f"Parsing error at row {idx}: {e}")
                result_to_save = output  # salva output grezzo

            row_result = {
                "model": "Gemini-2.5-Flash",
                "execution_type": EXEC_TYPE,
                "db_path": db_path,
                "table": table,
                "sql_query": sql,
                "question": prompt,
                "result": result_to_save
            }

            pd.DataFrame([row_result]).to_csv(out_path, mode='a', index=False, header=not os.path.exists(out_path))

            progress[test_path] = idx
            with open(progress_path, 'w') as f:
                json.dump(progress, f)

            time.sleep(10)  
            break  

        except Exception as e:
            error_msg = str(e)

            if "429" in error_msg and "quota" in error_msg.lower():
                print(f"Limite quota raggiunto. Attendo {attesa} ore\n")
                time.sleep(attesa * 60 * 60)  
                continue  
            else:
                break 

print("Output salvato in", out_path)

out_file = pd.read_csv(out_path)
compute_res.process_file(out_path)

