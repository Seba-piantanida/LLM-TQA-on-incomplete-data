import pandas as pd
import numpy as np
import sqlite3
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import LabelEncoder

def find_removable_columns(
    df,
    r2_threshold=0.8,
    accuracy_threshold=0.8,
    test_size=0.2,
    random_state=42,
):
    removable_columns = []
    data = df.copy()

    # Encoding per le colonne categoriche
    label_encoders = {}
    for col in data.select_dtypes(include=["object", "category"]).columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))
        label_encoders[col] = le

    columns = data.columns.tolist()

    for target_col in columns:
        feature_cols = [col for col in columns if col != target_col]

        # Se il target ha troppi NaN 
        if data[target_col].isna().mean() > 0.2:
            print(f"    ⚠️  Troppi NaN nella colonna '{target_col}', la salto.")
            continue

        # Rimuove righe con NaN
        subset = data[feature_cols + [target_col]].dropna()

        if subset.shape[0] < 10:
            print(f"    ⚠️  Troppi pochi dati validi per '{target_col}', la salto.")
            continue

        X = subset[feature_cols]
        y = subset[target_col]

        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        
        if pd.api.types.is_numeric_dtype(df[target_col]):
            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            score = r2_score(y_test, y_pred)
            metric = "R²"
        else:
            model = RandomForestClassifier(n_estimators=50, random_state=random_state)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            score = accuracy_score(y_test, y_pred)
            metric = "Accuracy"

        print(f"    Colonna '{target_col}': {metric} = {score:.3f}")

        
        if (pd.api.types.is_numeric_dtype(df[target_col]) and score >= r2_threshold) or \
           (not pd.api.types.is_numeric_dtype(df[target_col]) and score >= accuracy_threshold):
            removable_columns.append(target_col)

    return removable_columns

def load_all_tables_from_sqlite(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]

    conn.close()
    return tables

def load_dataframe_from_sqlite(db_path, table_name):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    conn.close()
    return df


if __name__ == "__main__":
    db_path = "data/db.sqlite" 

    # Trova tutte le tabelle
    tables = load_all_tables_from_sqlite(db_path)
    print(f"Trovate {len(tables)} tabelle nel database: {tables}")

    # Analizza ogni tabella
    for table_name in tables:
        print(f"\n🔍 Analisi tabella: {table_name}")

        df = load_dataframe_from_sqlite(db_path, table_name)

        if df.empty:
            print("    ⚠️  Tabella vuota, salto.")
            continue

        columns_to_remove = find_removable_columns(df)

        print(f"    ➔ Colonne suggerite per la rimozione in '{table_name}': {columns_to_remove}")