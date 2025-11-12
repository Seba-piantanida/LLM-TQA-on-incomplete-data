import pandas as pd
import sqlite3



def SQL_query(tests):

    
    answers = pd.DataFrame(columns=['query','SQL_answer'])
    df = pd.read_csv(tests)
    data_base = 'data/db_cut_300.sqlite'
    conn = sqlite3.connect(data_base)
    cursor = conn.cursor()
    for index, row in df.iterrows():
        query = row['query']  
        try:
            cursor.execute(query )
            results = cursor.fetchall()
            answers.loc[index] = [row['question'], [result for result in results]]
            
        except sqlite3.Error as e:
            print(f"Error executing query for row {index+1}: {e}")

    return answers

ai = pd.read_csv('output/API_test.csv')
sql = SQL_query('test.csv')

res: pd.DataFrame = pd.merge(ai, sql, how='left')
print(res)

res.to_csv('output/merged_res.csv', index=False)

