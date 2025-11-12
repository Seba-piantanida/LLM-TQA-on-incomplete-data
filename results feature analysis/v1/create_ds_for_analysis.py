import pandas as pd
import sqlite3
import sqlparse
import re
from pathlib import Path

AGG_FUNCTIONS = {"sum", "avg", "max", "min", "count", "total"}

def get_table_info(db_path, table_name):
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()
        col_names = [col[1] for col in columns]
        col_types = [col[2].upper() for col in columns]

        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        n_rows = cursor.fetchone()[0]
        conn.close()

        n_numeric = sum(1 for t in col_types if t in ['INTEGER', 'REAL', 'NUMERIC'])
        n_text = sum(1 for t in col_types if t == 'TEXT')

        return {
            "n_columns": len(columns),
            "n_rows": n_rows,
            "column_names": col_names,
            "column_types": col_types,
            "n_numeric_cols": n_numeric,
            "n_text_cols": n_text,
            "pct_numeric_cols": n_numeric / len(columns) if columns else 0,
            "pct_text_cols": n_text / len(columns) if columns else 0,
        }

    except Exception as e:
        return {
            "n_columns": None, "n_rows": None, "column_names": [],
            "column_types": [], "n_numeric_cols": None, "n_text_cols": None,
            "pct_numeric_cols": None, "pct_text_cols": None,
            "error": str(e)
        }

def parse_sql_clauses(query):
    query_lower = query.lower()
    return {
        "has_orderby": "order by" in query_lower,
        "has_limit": "limit" in query_lower,
        "has_where": "where" in query_lower,
        "has_join": " join " in query_lower,
        "has_groupby": "group by" in query_lower,
        "has_having": "having" in query_lower,
        "has_distinct": "distinct" in query_lower,
        "uses_arithmetic": any(op in query_lower for op in [" + ", " - ", " * ", " / "]),
        "uses_aggregation": any(func in query_lower for func in AGG_FUNCTIONS),
    }

def extract_query_features(query, table_cols, col_types):
    query_lower = query.lower()

    used_cols = [col for col in table_cols if re.search(rf"\b{re.escape(col.lower())}\b", query_lower)]
    types_used = set()
    for col in used_cols:
        if col in table_cols:
            idx = table_cols.index(col)
            types_used.add(col_types[idx])

    # Numero colonne in output (grezza ma utile)
    select_match = re.search(r'select (.+?) from', query_lower)
    if select_match:
        select_part = select_match.group(1)
        n_output_columns = len([s.strip() for s in select_part.split(',') if s.strip() != '*'])
        uses_star = '*' in select_part
    else:
        n_output_columns = 0
        uses_star = False

    clause_features = parse_sql_clauses(query)

    return {
        "columns_in_query": ', '.join(used_cols),
        "n_columns_in_query": len(used_cols),
        "column_types_used": ', '.join(types_used),
        "n_output_columns": n_output_columns,
        "uses_star": uses_star,
        **clause_features
    }

def enrich_dataset(df):
    db_cache = {}
    enriched_rows = []

    for _, row in df.iterrows():
        db_path = row["db_path"]
        table_name = row["table_name"]
        query = row["SQL_query"]

        db_key = (db_path, table_name)

        if db_key not in db_cache:
            db_cache[db_key] = get_table_info(db_path, table_name)

        table_info = db_cache[db_key]
        col_names = [c.lower() for c in table_info.get("column_names", [])]
        col_types = table_info.get("column_types", [])

        query_feats = extract_query_features(query, col_names, col_types)

        # Forzatura dei tipi numerici per colonne derivate
        enriched_row = {
            **row,
            "n_columns": int(table_info.get("n_columns") or 0),
            "n_rows": int(table_info.get("n_rows") or 0),
            "n_numeric_cols": int(table_info.get("n_numeric_cols") or 0),
            "n_text_cols": int(table_info.get("n_text_cols") or 0),
            "pct_numeric_cols": float(table_info.get("pct_numeric_cols") or 0.0),
            "pct_text_cols": float(table_info.get("pct_text_cols") or 0.0),
            "n_columns_in_query": int(query_feats.get("n_columns_in_query") or 0),
            "n_output_columns": int(query_feats.get("n_output_columns") or 0),
            "uses_star": int(query_feats.get("uses_star") or 0),
        }

        # Converte tutti i flag booleani in int (0 o 1)
        for key, value in query_feats.items():
            if isinstance(value, bool):
                enriched_row[key] = int(value)

        # Salviamo comunque anche le stringhe per analisi successive (verranno eventualmente rimosse in preprocess)
        enriched_row["columns_in_query"] = query_feats.get("columns_in_query", "")
        enriched_row["column_types_used"] = query_feats.get("column_types_used", "")

        enriched_rows.append(enriched_row)

    return pd.DataFrame(enriched_rows)
# === USO ===
input_csv = "out_test_simple_sum_normal_QATCH.csv"
output_csv = "enriched_simple_tests.csv"

df = pd.read_csv(input_csv)
enriched_df = enrich_dataset(df)

# Calcolo mean_score tra le metriche indicate
metrics_to_average = [
    "cell_precision", "cell_recall", "execution_accuracy",
    "tuple_cardinality", "tuple_constraint", "tuple_order"
]

# Verifica che le metriche esistano e calcola media solo su quelle presenti
existing_metrics = [col for col in metrics_to_average if col in enriched_df.columns]
enriched_df["mean_score"] = enriched_df[existing_metrics].mean(axis=1)

# Salvataggio
enriched_df.to_csv(output_csv, index=False)
print(f"✅ File salvato in: {output_csv}")