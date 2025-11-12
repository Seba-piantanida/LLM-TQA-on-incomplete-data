import pandas as pd
import sqlite3
import re
import json
from playwright.sync_api import sync_playwright
import time
import os
import random
from qatch.evaluate_dataset.orchestrator_evaluator import OrchestratorEvaluator
import compute_res

model = 'gpt-5_mini'
test_file = 'numerical_simple'
test_path = f'tests/test_{test_file}.csv'
EXEC_TYPE = 'NORMAL'  # 'REMOVE', 'NULL', 'NORMAL'
out_path = f'output/type_fix/{test_file}/{test_file}_type_fix_{EXEC_TYPE}_{model}.csv'

progress_path = 'progress_gpt.json'



if os.path.exists(progress_path):
    with open(progress_path, 'r') as f:
        progress = json.load(f)
else:
    progress = {}

last_completed_idx = int(progress.get(test_path, -1))


df = pd.read_csv(test_path, dtype=str).dropna(subset=['query', 'question'])
df = df[df.index > last_completed_idx]

append_mode = 'a'
header_needed = not os.path.exists(out_path)


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

import json
import re

def extract_json(output):
    output = output.strip()

    
    code_block_match = re.search(r'```(?:json)?\s*([\s\S]+?)\s*```', output)
    if code_block_match:
        json_str = code_block_match.group(1)
    else:
        json_match = re.search(r'(\{[\s\S]+?\})', output)
        json_str = json_match.group(1) if json_match else output

    json_str = re.sub(r'^(json|jsonCopiaModifica)\s*', '', json_str).strip()

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

        elif EXEC_TYPE == 'NULL':
            if column in df.columns:
                df[column] = None

    return f"{table_name}: {df.to_json(index=False)}\n"
results = []

with sync_playwright() as p:
    chrome_path = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

    browser = p.chromium.launch_persistent_context(
        user_data_dir="/Users/seba/Library/Application Support/Google/Chrome/Profile 5", 
        executable_path=chrome_path,
        headless=False,
        args=[
            "--disable-blink-features=AutomationControlled",
            "--start-maximized"
        ]
    )

    page = browser.new_page()
    page.goto("https://chat.openai.com")
   
    input("Press any key after logging in")

    
    system_msg = """You are a highly skilled data analyst. Always follow instructions carefully. Always respond strictly in valid JSON."""
    input_box = page.locator('div.ProseMirror[contenteditable="true"]')
    input_box.wait_for(state="visible", timeout=60000)

    #send system message
    input_box.click()
    input_box.type(system_msg, delay=50) 
    input_box.press("Enter")
    time.sleep(5)
    

    for idx, row in df.iterrows():
        
        try:
            db_path = row['db_path']
            table = row['tbl_name']
            sql = row['query']
            prompt = row['question']
            dataset_json = get_dataset(table, db_path, sql)

            full_prompt = f"""
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

            #send prompt
            textarea = page.locator('div.ProseMirror[contenteditable="true"]')
            textarea.fill(full_prompt)
            textarea.press("Enter")
            time.sleep(random.uniform(35,47)) 
            # wait response
            page.wait_for_selector("div[data-message-author-role='assistant']", timeout=90000)
            time.sleep(15)
            responses = page.locator("div[data-message-author-role='assistant']")
            last_response = responses.last.text_content()

            #extract JSON from response
            element = page.locator("div[data-message-author-role='assistant'] code").last
            json_text = element.evaluate("el => el.textContent")
            json_text = re.sub(r'^(json|jsonCopiaModifica)?\s*', '', json_text).strip()

            try:
                parsed = json.loads(json_text)

               
                if "ordered_entries" in parsed and isinstance(parsed["ordered_entries"], list):
                    entries = parsed["ordered_entries"]
                    array_of_arrays = [[entry[key] for key in entry] for entry in entries]
                    result_to_save = array_of_arrays
                else:
                    raise ValueError("'ordered_entries' missing")
            except Exception as e:
                print(f"error on line {idx}: {e}")
                result_to_save = json_text  # salviamo comunque l'output grezzo e pulito

            row_result = {
                "model": model,
                "execution_type": EXEC_TYPE,
                "db_path": db_path,
                "table": table,
                "sql_query": sql,
                "question": prompt,
                "result": result_to_save
            }

            # Salva riga in CSV (append)
            pd.DataFrame([row_result]).to_csv(out_path, mode='a', index=False, header=not os.path.exists(out_path))

            # Aggiorna il file di progresso
            progress[test_path] = idx
            with open(progress_path, 'w') as f:
                json.dump(progress, f)


            

        except Exception as e:
            print(f"Errore con riga {idx}: {e}")
            continue

    browser.close()


print("Output salvato in ", out_path)
out_file = pd.read_csv(out_path)
compute_res.process_file(out_path)