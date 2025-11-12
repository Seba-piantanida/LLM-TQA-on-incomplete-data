import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model  
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import sqlite3
import json


def build_autoencoder(input_dim):
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(64, activation='relu')(input_layer)
    encoded = Dense(32, activation='relu')(encoded)
    decoded = Dense(64, activation='relu')(encoded)
    decoded = Dense(input_dim, activation='sigmoid')(decoded)
    autoencoder = Model(input_layer, decoded)
    encoder = Model(input_layer, encoded)
    
    autoencoder.compile(optimizer=Adam(), loss='mean_squared_error')
    return autoencoder


def calculate_reconstruction_errors(df):
    errors = {}
    
    # Separiamo le colonne numeriche e non numeriche
    numeric_df = df.select_dtypes(include=[np.number])
    categorical_df = df.select_dtypes(exclude=[np.number])
    
    # Se ci sono colonne numeriche
    if not numeric_df.empty:
        # Rimuoviamo righe con NaN nelle colonne numeriche
        numeric_df = numeric_df.dropna(axis=0)
        
        # Normalizziamo i dati numerici
        scaler = StandardScaler()
        normalized_data = scaler.fit_transform(numeric_df)
        
        for col in numeric_df.columns:
            X = normalized_data.copy()
            y = X[:, numeric_df.columns.get_loc(col)]  # Colonna da predire
            
            # Addestra l'autoencoder
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            autoencoder = build_autoencoder(X_train.shape[1])
            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            autoencoder.fit(X_train, X_train, epochs=50, batch_size=32, validation_data=(X_test, X_test), callbacks=[early_stopping], verbose=0)
            
            # Calcola l'errore di ricostruzione
            reconstructed = autoencoder.predict(X_test)
            mse = mean_squared_error(X_test, reconstructed)
            errors[col] = mse  # Salva l'errore per la colonna
    
    # Per le colonne categoriche
    if not categorical_df.empty:
        # Rimuoviamo righe con NaN nelle colonne categoriche
        categorical_df = categorical_df.dropna(axis=0)
        
        # Codifica OneHot per le colonne categoriche e converte in array denso
        encoder = OneHotEncoder()
        encoded_data = encoder.fit_transform(categorical_df).toarray()  # Converte in array denso
        
        for i, col in enumerate(categorical_df.columns):
            X = encoded_data.copy()
            y = X[:, i]  # Colonna da predire
            
            # Addestra l'autoencoder
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            autoencoder = build_autoencoder(X_train.shape[1])
            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            autoencoder.fit(X_train, X_train, epochs=50, batch_size=32, validation_data=(X_test, X_test), callbacks=[early_stopping], verbose=0)
            
            # Calcola l'errore di ricostruzione
            reconstructed = autoencoder.predict(X_test)
            mse = mean_squared_error(X_test, reconstructed)
            errors[col] = mse  # Salva l'errore per la colonna

    return errors

# Carica il dataset (assicurati di avere un DB SQLite già configurato)
def load_data_from_sqlite(db_path, table_name):
    conn = sqlite3.connect(db_path)
    query = f"SELECT * FROM {table_name}"
    df = pd.read_sql(query, conn)
    conn.close()
    return df

# Salva i risultati in un file JSON
def save_results_to_json(all_results, filename='remuvable_colums.json'):
    with open(filename, 'w') as json_file:
        json.dump(all_results, json_file, indent=4)

db_path = 'data/db.sqlite'  
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = [row[0] for row in cursor.fetchall()]

all_results = {}

for table in tables:
    print(f"🔍 Analisi tabella: {table}")
    df = load_data_from_sqlite(db_path, table)
    errors = calculate_reconstruction_errors(df)
    
    if not errors:
        print(f"    ⚠️ Nessuna colonna numerica valida trovata in '{table}'")
        print("\n")
        continue
    
    # Stampa gli errori per ogni colonna
    for col, error in errors.items():
        print(f"    Colonna '{col}': Errore ricostruzione = {error:.4f}")
    
    # Decidi quali colonne rimuovere (soglia a piacere)
    removable_columns = [col for col, error in errors.items() if error < 0.15]  # Modifica la soglia come preferisci
    print(f"    ➔ Colonne suggerite per la rimozione in '{table}': {removable_columns}")
    print("\n")
    
    # Aggiungi i risultati al dizionario globale
    all_results[table] = {
        'removable_columns': removable_columns,
        'reconstruction_errors': errors
    }

# Salva tutti i risultati in un file JSON
save_results_to_json(all_results)

print("I risultati sono stati salvati in 'reconstruction_errors.json'.")