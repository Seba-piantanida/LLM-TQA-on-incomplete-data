import sqlite3
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

def analyze_table_inferibility(df, score_threshold=0.85, cv=5):
    results = []

    df = df.dropna()  # semplificazione: ignora righe con NaN

    for target_col in df.columns:
        X = df.drop(columns=[target_col])
        y = df[target_col]

        # One-hot encoding su X
        X_encoded = pd.get_dummies(X)

        if y.dtype == 'object' or y.nunique() < 20:
            y_encoded = LabelEncoder().fit_transform(y)
            model = RandomForestClassifier(n_estimators=100)
            scoring = 'accuracy'
        else:
            y_encoded = y
            model = RandomForestRegressor(n_estimators=100)
            scoring = 'r2'

        try:
            score = cross_val_score(model, X_encoded, y_encoded, scoring=scoring, cv=cv).mean()
        except:
            score = np.nan

        results.append({
            'colonna': target_col,
            'tipo': 'categorica' if scoring == 'accuracy' else 'numerica',
            'metrica': scoring,
            'score': round(score, 3) if not np.isnan(score) else None,
            'inferibile': score >= score_threshold if not np.isnan(score) else None
        })

    return pd.DataFrame(results)

def analyze_sqlite_db(db_path, score_threshold=0.85):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]
    
    all_reports = {}

    for table in tables:
        print(f"Analizzando tabella: {table}")
        df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
        report = analyze_table_inferibility(df, score_threshold=score_threshold)
        all_reports[table] = report

    conn.close()
    return all_reports

DB_PATH = 'data/db.sqlite'
report_per_tabella = analyze_sqlite_db(DB_PATH)

for tab in report_per_tabella:
    print(report_per_tabella[tab])