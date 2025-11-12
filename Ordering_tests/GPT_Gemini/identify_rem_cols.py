import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model  
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import sqlite3
import json
import warnings
warnings.filterwarnings('ignore')


def build_column_autoencoder(input_dim, target_col_size=1):
    """
    Costruisce un autoencoder specifico per ricostruire una singola colonna
    """
    # Input layer (tutte le colonne tranne quella target)
    input_layer = Input(shape=(input_dim,))
    
    # Encoder
    encoded = Dense(128, activation='relu')(input_layer)
    encoded = Dropout(0.2)(encoded)
    encoded = Dense(64, activation='relu')(encoded)
    encoded = Dropout(0.2)(encoded)
    encoded = Dense(32, activation='relu')(encoded)
    
    # Decoder per ricostruire solo la colonna target
    decoded = Dense(64, activation='relu')(encoded)
    decoded = Dropout(0.2)(decoded)
    decoded = Dense(32, activation='relu')(decoded)
    output = Dense(target_col_size, activation='linear')(decoded)  # Linear per valori continui
    
    autoencoder = Model(input_layer, output)
    autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    
    return autoencoder


def build_categorical_autoencoder(input_dim, target_categories):
    """
    Costruisce un autoencoder specifico per colonne categoriche
    """
    input_layer = Input(shape=(input_dim,))
    
    # Encoder
    encoded = Dense(128, activation='relu')(input_layer)
    encoded = Dropout(0.2)(encoded)
    encoded = Dense(64, activation='relu')(encoded)
    encoded = Dropout(0.2)(encoded)
    encoded = Dense(32, activation='relu')(encoded)
    
    # Decoder
    decoded = Dense(64, activation='relu')(encoded)
    decoded = Dropout(0.2)(decoded)
    decoded = Dense(32, activation='relu')(decoded)
    output = Dense(target_categories, activation='softmax')(decoded)  # Softmax per categorie
    
    autoencoder = Model(input_layer, output)
    autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return autoencoder


def calculate_reconstruction_errors(df, min_samples=50):
    """
    Calcola gli errori di ricostruzione per ogni colonna rimuovendola sistematicamente
    """
    errors = {}
    
    # Rimuovi righe con troppi valori mancanti
    df_clean = df.dropna(thresh=len(df.columns)*0.7)  # Mantieni righe con almeno 70% di valori
    
    if len(df_clean) < min_samples:
        print(f"    ⚠️ Campioni insufficienti dopo la pulizia: {len(df_clean)} < {min_samples}")
        return errors
    
    # Separa colonne numeriche e categoriche
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df_clean.select_dtypes(exclude=[np.number]).columns.tolist()
    
    # Processa colonne numeriche
    for target_col in numeric_cols:
        try:
            error = process_numeric_column(df_clean, target_col, numeric_cols, categorical_cols)
            if error is not None:
                errors[target_col] = error
        except Exception as e:
            print(f"    ⚠️ Errore con colonna numerica '{target_col}': {str(e)}")
            continue
    
    # Processa colonne categoriche
    for target_col in categorical_cols:
        try:
            error = process_categorical_column(df_clean, target_col, numeric_cols, categorical_cols)
            if error is not None:
                errors[target_col] = error
        except Exception as e:
            print(f"    ⚠️ Errore con colonna categorica '{target_col}': {str(e)}")
            continue
    
    return errors


def process_numeric_column(df_clean, target_col, numeric_cols, categorical_cols):
    """
    Processa una colonna numerica per calcolare l'errore di ricostruzione
    """
    # Crea dataset senza la colonna target
    input_cols = [col for col in numeric_cols if col != target_col]
    
    if len(input_cols) == 0:
        return None
    
    # Filtra righe senza valori mancanti per target e input
    mask = df_clean[input_cols + [target_col]].notna().all(axis=1)
    data_subset = df_clean[mask]
    
    if len(data_subset) < 30:  # Numero minimo di campioni
        return None
    
    # Prepara i dati
    X_input = data_subset[input_cols].values
    y_target = data_subset[target_col].values.reshape(-1, 1)
    
    # Normalizza i dati di input
    scaler_input = StandardScaler()
    X_input_scaled = scaler_input.fit_transform(X_input)
    
    # Normalizza il target
    scaler_target = StandardScaler()
    y_target_scaled = scaler_target.fit_transform(y_target)
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_input_scaled, y_target_scaled, test_size=0.3, random_state=42
    )
    
    # Costruisci e addestra l'autoencoder
    autoencoder = build_column_autoencoder(X_train.shape[1], target_col_size=1)
    
    early_stopping = EarlyStopping(
        monitor='val_loss', 
        patience=15, 
        restore_best_weights=True,
        min_delta=1e-4
    )
    
    autoencoder.fit(
        X_train, y_train, 
        epochs=100, 
        batch_size=min(32, len(X_train)//4), 
        validation_data=(X_test, y_test), 
        callbacks=[early_stopping], 
        verbose=0
    )
    
    # Calcola l'errore di ricostruzione
    y_pred_scaled = autoencoder.predict(X_test, verbose=0)
    y_pred = scaler_target.inverse_transform(y_pred_scaled)
    y_test_original = scaler_target.inverse_transform(y_test)
    
    mse_error = mean_squared_error(y_test_original, y_pred)
    mae_error = mean_absolute_error(y_test_original, y_pred)
    
    # Normalizza l'errore rispetto alla varianza del target
    target_var = np.var(y_test_original)
    normalized_error = mse_error / (target_var + 1e-8)
    
    return normalized_error


def process_categorical_column(df_clean, target_col, numeric_cols, categorical_cols):
    """
    Processa una colonna categorica per calcolare l'errore di ricostruzione
    """
    # Crea dataset senza la colonna target
    input_cols = [col for col in numeric_cols + categorical_cols if col != target_col]
    
    if len(input_cols) == 0:
        return None
    
    # Filtra righe senza valori mancanti
    mask = df_clean[input_cols + [target_col]].notna().all(axis=1)
    data_subset = df_clean[mask]
    
    if len(data_subset) < 30:
        return None
    
    # Codifica la colonna target
    label_encoder = LabelEncoder()
    y_target = label_encoder.fit_transform(data_subset[target_col].values)
    n_classes = len(label_encoder.classes_)
    
    if n_classes < 2:  # Almeno 2 classi per essere significativo
        return None
    
    # Prepara le colonne di input
    X_input = []
    scalers = []
    
    for col in input_cols:
        if col in numeric_cols:
            values = data_subset[col].values.reshape(-1, 1)
            scaler = StandardScaler()
            scaled_values = scaler.fit_transform(values)
            X_input.append(scaled_values)
            scalers.append(scaler)
        else:
            # Per colonne categoriche, usa label encoding
            encoder = LabelEncoder()
            encoded_values = encoder.fit_transform(data_subset[col].values).reshape(-1, 1)
            # Normalizza anche gli encoding categorici
            encoded_values = encoded_values / (len(encoder.classes_) - 1)
            X_input.append(encoded_values)
            scalers.append(encoder)
    
    if len(X_input) == 0:
        return None
        
    X_input = np.hstack(X_input)
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_input, y_target, test_size=0.3, random_state=42, stratify=y_target
    )
    
    # Costruisci e addestra l'autoencoder
    autoencoder = build_categorical_autoencoder(X_train.shape[1], n_classes)
    
    early_stopping = EarlyStopping(
        monitor='val_loss', 
        patience=15, 
        restore_best_weights=True,
        min_delta=1e-4
    )
    
    autoencoder.fit(
        X_train, y_train, 
        epochs=100, 
        batch_size=min(32, len(X_train)//4), 
        validation_data=(X_test, y_test), 
        callbacks=[early_stopping], 
        verbose=0
    )
    
    # Calcola l'accuracy come metrica di ricostruzione
    _, accuracy = autoencoder.evaluate(X_test, y_test, verbose=0)
    
    # Converti accuracy in errore (1 - accuracy)
    reconstruction_error = 1.0 - accuracy
    
    return reconstruction_error


def load_data_from_sqlite(db_path, table_name):
    conn = sqlite3.connect(db_path)
    query = f"SELECT * FROM {table_name}"
    df = pd.read_sql(query, conn)
    conn.close()
    return df


def save_results_to_json(all_results, filename='reconstruction_errors.json'):
    with open(filename, 'w') as json_file:
        json.dump(all_results, json_file, indent=4)


# Script principale
db_path = 'data/db.sqlite'  
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = [row[0] for row in cursor.fetchall()]

all_results = {}

for table in tables:
    print(f"🔍 Analisi tabella: {table}")
    df = load_data_from_sqlite(db_path, table)
    
    print(f"    📊 Dimensioni dataset: {df.shape}")
    print(f"    📈 Colonne numeriche: {len(df.select_dtypes(include=[np.number]).columns)}")
    print(f"    📝 Colonne categoriche: {len(df.select_dtypes(exclude=[np.number]).columns)}")
    
    errors = calculate_reconstruction_errors(df)
    
    if not errors:
        print(f"    ⚠️ Nessuna colonna processabile in '{table}'")
        print("\n")
        continue
    
    # Stampa gli errori per ogni colonna
    for col, error in sorted(errors.items(), key=lambda x: x[1]):
        print(f"    📋 Colonna '{col}': Errore ricostruzione = {error:.6f}")
    
    # Soglia dinamica basata sulla mediana degli errori
    error_values = list(errors.values())
    median_error = np.median(error_values)
    threshold = median_error * 0.5  
    
    removable_columns = [col for col, error in errors.items() if error < threshold]
    
    print(f"    🎯 Soglia calcolata: {threshold:.6f}")
    print(f"    ➔ Colonne suggerite per la rimozione in '{table}': {removable_columns}")
    print("\n")
    
    
    
    all_results[table] = {
        'removable_columns': removable_columns,
        'reconstruction_errors': errors,
        'threshold_used': threshold,
        'median_error': median_error
    }


save_results_to_json(all_results)
print("✅ I risultati sono stati salvati in 'reconstruction_errors.json'.")


total_tables = len(all_results)
total_removable = sum(len(result['removable_columns']) for result in all_results.values())

print(f"\n📊 RIASSUNTO FINALE:")
print(f"   Tabelle analizzate: {total_tables}")
print(f"   Colonne totali identificate come rimuovibili: {total_removable}")
