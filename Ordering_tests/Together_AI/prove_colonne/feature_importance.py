import pandas as pd
import numpy as np
from sqlalchemy import create_engine, inspect
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer


DB_URL = 'sqlite:///data/db.sqlite'  

engine = create_engine(DB_URL)
inspector = inspect(engine)
table_names = inspector.get_table_names()

def preprocess_dataframe(df):
    df_clean = df.copy()
    for col in df_clean.select_dtypes(include='object'):
        df_clean[col] = df_clean[col].astype(str).fillna('missing')
        df_clean[col] = LabelEncoder().fit_transform(df_clean[col])
    imputer = SimpleImputer(strategy='most_frequent')
    df_clean = pd.DataFrame(imputer.fit_transform(df_clean), columns=df.columns)
    return df_clean

def evaluate_column_deducibility(df, threshold_class=0.85, threshold_reg=0.8):
    df = preprocess_dataframe(df)
    deducible_columns = []
    scores = {}

    for target_col in df.columns:
        X = df.drop(columns=[target_col])
        y = df[target_col]

        if pd.api.types.is_numeric_dtype(y) and len(np.unique(y)) > 10:
            model = RandomForestRegressor()
            scoring = 'r2'
            threshold = threshold_reg
        else:
            model = RandomForestClassifier()
            scoring = 'accuracy'
            threshold = threshold_class

        try:
            score = cross_val_score(model, X, y, cv=5, scoring=scoring).mean()
        except:
            score = np.nan

        scores[target_col] = score
        if score >= threshold:
            deducible_columns.append((target_col, score))

    return deducible_columns, scores


for table in table_names:
    print(f"\n🗂️  Analisi della tabella: {table}")
    try:
        df = pd.read_sql_table(table, con=engine)
        if df.empty or df.shape[1] < 2:
            print("Tabella vuota o con meno di 2 colonne, salto.")
            continue

        deducibili, score_totali = evaluate_column_deducibility(df)

        if deducibili:
            print("Colonne deducibili:")
            for col, score in deducibili:
                print(f"  - {col}: {score:.2f}")
        else:
            print("Nessuna colonna deducibile trovata.")
    except Exception as e:
        print(f"Errore nella lettura/analisi della tabella '{table}': {e}")