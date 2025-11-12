import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import sqlite3
import json

def build_autoencoder(input_dim, encoding_dim=32):
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(64, activation='relu')(input_layer)
    encoded = Dense(encoding_dim, activation='relu')(encoded)
    decoded = Dense(64, activation='relu')(encoded)
    decoded = Dense(input_dim, activation='linear')(decoded)
    autoencoder = Model(input_layer, decoded)
    encoder = Model(input_layer, encoded)
    autoencoder.compile(optimizer=Adam(), loss='mse')
    return autoencoder, encoder

def encode_dataframe(df):
    df = df.copy()
    numeric = df.select_dtypes(include=[np.number])
    categorical = df.select_dtypes(exclude=[np.number])

    # One-hot encode categoricals
    if not categorical.empty:
        categorical = categorical.fillna("missing").astype(str)
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        categorical_encoded = encoder.fit_transform(categorical)
    else:
        categorical_encoded = np.empty((len(df), 0))

    # Scale numeric
    if not numeric.empty:
        numeric = numeric.fillna(numeric.mean())
        scaler = StandardScaler()
        numeric_scaled = scaler.fit_transform(numeric)
    else:
        numeric_scaled = np.empty((len(df), 0))

    combined = np.hstack([numeric_scaled, categorical_encoded])
    return combined

def evaluate_column_predictability(df, threshold_r2=0.8, threshold_acc=0.85):
    results = {}
    df = df.dropna(axis=0)  # Semplice pulizia (puoi migliorare)

    for col in df.columns:
        print(f"➡️  Valuto la colonna: {col}")
        try:
            y = df[col]
            X_df = df.drop(columns=[col])

            X_encoded = encode_dataframe(X_df)
            if X_encoded.shape[1] < 1:
                print("    ❌ Dataset troppo piccolo dopo la rimozione.")
                continue

            # Step 1: autoencoder
            X_train, X_test = train_test_split(X_encoded, test_size=0.2, random_state=42)
            autoencoder, encoder = build_autoencoder(X_train.shape[1])
            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            autoencoder.fit(X_train, X_train, epochs=50, batch_size=32,
                            validation_data=(X_test, X_test), callbacks=[early_stopping], verbose=0)

            X_encoded_train = encoder.predict(X_train)
            X_encoded_test = encoder.predict(X_test)

            # Step 2: supervised model
            if pd.api.types.is_numeric_dtype(y) and y.nunique() > 10:
                model = RandomForestRegressor()
                y_train, y_test = train_test_split(y, test_size=0.2, random_state=42)
                model.fit(X_encoded_train, y_train)
                y_pred = model.predict(X_encoded_test)
                score = r2_score(y_test, y_pred)
                metric = "R2"
                deducible = score >= threshold_r2
            else:
                le = LabelEncoder()
                y_enc = le.fit_transform(y)
                y_train, y_test = train_test_split(y_enc, test_size=0.2, random_state=42)
                model = RandomForestClassifier()
                model.fit(X_encoded_train, y_train)
                y_pred = model.predict(X_encoded_test)
                score = accuracy_score(y_test, y_pred)
                metric = "accuracy"
                deducible = score >= threshold_acc

            print(f"    ✅ Score {metric}: {score:.3f} → {'deducibile' if deducible else 'non deducibile'}")
            results[col] = {
                "score": float(score),
                "metric": metric,
                "deducible": deducible
            }

        except Exception as e:
            print(f"    ⚠️ Errore nella valutazione: {e}")
            results[col] = {
                "error": str(e),
                "deducible": False
            }

    return results

# === DB LOOP ===
def load_all_tables(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]
    conn.close()
    return tables

def load_data_from_sqlite(db_path, table_name):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
    conn.close()
    return df

import numpy as np

def convert_numpy_types(obj):
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(v) for v in obj]
    elif isinstance(obj, np.generic):
        return obj.item()  # converte numpy types (int64, float32, bool_) in tipi Python
    else:
        return obj

def save_results_to_json(results, filename='rem_columsV2.json'):
    clean_results = convert_numpy_types(results)
    with open(filename, 'w') as f:
        json.dump(clean_results, f, indent=4)

# === MAIN ===
db_path = 'data/db.sqlite'
all_tables = load_all_tables(db_path)
final_results = {}

for table in all_tables:
    print(f"\n📊 Analisi della tabella: {table}")
    df = load_data_from_sqlite(db_path, table)
    if df.shape[1] < 2:
        print("    ⚠️ Tabella troppo piccola, salto.")
        continue

    results = evaluate_column_predictability(df)
    final_results[table] = results

save_results_to_json(final_results)
print("\n✅ Risultati salvati in 'deducibility_results.json'.")