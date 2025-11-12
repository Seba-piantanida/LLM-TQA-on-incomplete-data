import pandas as pd
import numpy as np
import json
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

        # Inverti il punteggio per avere reconstruction error (minore è meglio)
        scores[target_col] = round(1 - score, 6)
        if score >= threshold:
            deducible_columns.append(target_col)

    return deducible_columns, scores

result_dict = {}

for table in table_names:
    try:
        df = pd.read_sql_table(table, con=engine)
        if df.empty or df.shape[1] < 2:
            continue

        deducible_columns, reconstruction_errors = evaluate_column_deducibility(df)

        result_dict[table] = {
            "removable_columns": deducible_columns,
            "reconstruction_errors": reconstruction_errors
        }
    except Exception as e:
        result_dict[table] = {
            "error": str(e)
        }

# Salvataggio JSON
with open("column_deducibility_results.json", "w") as f:
    json.dump(result_dict, f, indent=4)