# enrich_csv.py

import pandas as pd
import sqlite3
import re
import os
import numpy as np
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

nlp = spacy.load("en_core_web_sm")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
pca = PCA(n_components=1) 

AGG_FUNCTIONS = {"sum", "avg", "max", "min", "count", "total"}

def is_numeric_type(sql_type):
    if sql_type is None:
        return False
    return any(kw in sql_type.lower() for kw in ["int", "real", "double", "float", "decimal", "numeric"])

def analyze_table(db_path, table_name):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = cursor.fetchall()
    col_names = [col[1] for col in columns]
    col_types = [col[2] for col in columns]

    df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    conn.close()

    numeric_cols = [c for c, t in zip(col_names, col_types) if is_numeric_type(t)]
    categorical_cols = [c for c in col_names if c not in numeric_cols]

    return {
        "num_columns": len(col_names),
        "num_rows": len(df),
        "num_numeric_columns": len(numeric_cols),
        "num_categorical_columns": len(categorical_cols),
        "numeric_categorical_ratio": (len(numeric_cols) / (len(categorical_cols) + 1e-6)) if len(categorical_cols) != 0 else len(numeric_cols),
        "avg_unique_vals": df.nunique().mean(),
        "columns": col_names,
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols
    }

def analyze_query(sql_query, table_info):
    sql = sql_query.lower()
    if "select *" in sql:
        num_selected = table_info["num_columns"]
    else:
        select_match = re.search(r"select (.*?) from", sql, re.DOTALL)
        if select_match:
            selected = select_match.group(1)
            num_selected = len([s.strip() for s in selected.split(",")])
        else:
            num_selected = 0

    order_match = re.search(r"order by (.*?)(limit|$)", sql)
    if order_match:
        order_expr = order_match.group(1)
        order_cols = re.split(r"[\+\-*/]", order_expr)
        order_cols = [col.strip().strip("`") for col in order_cols if col.strip()]
    else:
        order_cols = []

    numeric_in_query = len([c for c in order_cols if c in table_info["numeric_columns"]])
    categorical_in_query = len([c for c in order_cols if c in table_info["categorical_columns"]])

    return {
        "num_selected_columns": num_selected,
        "num_order_columns": len(order_cols),
        "num_numeric_in_query": numeric_in_query,
        "num_categorical_in_query": categorical_in_query,
    }


def query_complexity_score(text, df):
    doc = nlp(text)

    # Feature linguistiche
    num_tokens = len(doc)
    num_nouns = len([t for t in doc if t.pos_ == "NOUN"])
    num_verbs = len([t for t in doc if t.pos_ == "VERB"])
    num_clauses = len([t for t in doc if t.dep_ in ("advcl", "relcl", "ccomp", "xcomp")])
    text_length = len(text)

    # Heuristica: parole chiave semantiche in NL query
    keywords = [
        "group", "order", "sum", "average", "filter", "sorted", "by", "top", "limit", "most", "least",
        "more than", "less than", "compared to", "difference", "total", "change", "per", "each", "across"
    ]
    keyword_score = sum(kw in text.lower() for kw in keywords)

    all_embeddings = embed_model.encode(df["query"].tolist(), show_progress_bar=True)
    pca = PCA(n_components=1)
    pca.fit(all_embeddings)

    # Embedding + PCA per indicare semantica densa
    emb = embed_model.encode([text])
    sem_complexity = float(pca.transform(emb)[0][0])

    # Punteggio aggregato (puoi modificarlo se vuoi pesare diversamente)
    agg_score = (
        0.2 * num_tokens +
        0.1 * num_nouns +
        0.1 * num_verbs +
        0.05 * text_length +
        0.15 * num_clauses +
        0.1 * keyword_score +
        0.3 * sem_complexity
    )
    return agg_score

def main(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    new_features = []

    for i, row in df.iterrows():
        try:
            table_info = analyze_table(row["db_path"], row["table_name"])
            query_info = analyze_query(row["SQL_query"], table_info)
            complexity = query_complexity_score(row["query"], df)
            combined = {
                **table_info,
                **query_info,
                "query_complexity": complexity
            }
        except Exception as e:
            print(f"Errore alla riga {i}: {e}")
            combined = {k: None for k in [
                "num_columns", "num_rows", "num_numeric_columns", "num_categorical_columns",
                "numeric_categorical_ratio", "avg_unique_vals", "num_selected_columns",
                "num_order_columns", "num_numeric_in_query", "num_categorical_in_query",
                "query_complexity"
            ]}

        new_features.append(combined)

    features_df = pd.DataFrame(new_features)
    result = pd.concat([df, features_df], axis=1)
    result.to_csv(output_csv, index=False)
    print(f"✅ CSV arricchito salvato in: {output_csv}")

if __name__ == "__main__":
    import sys
    input_csv = "v2/all_tests.csv"
    output_csv = f"{input_csv}_enriched.csv"
    main(input_csv, output_csv)