import pandas as pd
import sqlite3
import re
import json
import os
import time
import sys
from dotenv import load_dotenv
from datetime import datetime
import traceback
import numpy as np
from typing import List, Dict, Tuple, Optional
from scipy.stats import spearmanr
from together import Together

# === CONFIG ===
TOGETHER_MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo"  # Default model
load_dotenv()
API_KEY = os.getenv("TOGETHER_API_KEY")

time_out = 600  # tempo massimo attesa risposta
MAX_RETRIES = 3  # numero massimo di retry per ogni test
RETRY_DELAY = 40  # secondi di attesa tra i retry
QUOTA_WAIT_HOURS = 8  # ore di attesa per quota exceeded

# Variabile globale per le colonne da rimuovere/nullificare
rem_columns = ['title', 'year']

# === INIT TOGETHER AI ===
together_client = Together(api_key=API_KEY)

def log(msg):
    """Stampa messaggi con timestamp e forza il flush per nohup."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}")
    sys.stdout.flush()

def ensure_parent_dir(path: str):
    """Crea la cartella padre se non esiste."""
    if path is None or path.strip() == "":
        return
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

def load_dataset(csv_path: str, exec_type: str = "NORMAL"):
    """
    Carica il dataset dal CSV.

    Args:
        csv_path: path al file CSV del dataset
        exec_type: modalità di esecuzione ("NORMAL", "NULL", "REMOVE")

    Returns:
        pandas.DataFrame: dataset modificato secondo exec_type
    """
    try:
        log(f"📂 Caricamento dataset da {csv_path}")
        df = pd.read_csv(csv_path)

        if exec_type.upper() == "REMOVE":
            # Rimuove completamente le colonne specificate
            for col in rem_columns:
                if col in df.columns:
                    df = df.drop(columns=[col], errors='ignore')
                    log(f"❌ Rimossa colonna '{col}' dal dataset")

        elif exec_type.upper() == "NULL":
            # Setta a NULL le colonne specificate
            for col in rem_columns:
                if col in df.columns:
                    df[col] = None
                    log(f"🚫 Impostata colonna '{col}' a NULL nel dataset")

        elif exec_type.upper() == "NORMAL":
            log(f"✅ Nessuna modifica applicata al dataset")

        return df

    except Exception as e:
        log(f"⚠️ Errore nel caricamento del dataset {csv_path}: {e}")
        return None

def format_dataset_for_prompt(df: pd.DataFrame) -> str:
    """
    Formatta il dataset per il prompt di Together AI.

    Args:
        df: DataFrame del dataset

    Returns:
        str: dataset formattato come stringa JSON
    """
    return df.to_json(orient='records', indent=2)

def create_prompt(dataset_str: str, nl_query: str) -> str:
    """
    Crea il prompt per l'API di Together AI.

    Args:
        dataset_str: dataset formattato come stringa
        nl_query: query in linguaggio naturale

    Returns:
        str: prompt completo per Together AI
    """
    prompt = f"""
You are a movie recommendation system. Analyze the provided movie dataset and respond to the user's request.

Dataset:
{dataset_str}

User Request:
{nl_query}

Instructions:
- Analyze the movies in the dataset
- For similarity requests, find movies with similar genres, themes, or characteristics
- For negative similarity requests ("opposto", "opposite", "contrario"), find movies that are opposite in style, genre, or theme
- Return the results as a list of movie IDs (tt codes) that best match the request
- Consider factors like genre, year, rating, plot, and other available attributes when available
- Provide exactly 10 recommendations when possible
- If some data is missing (NULL values), use the available information to make the best recommendations

Response Format:
Return only the movie IDs (tt codes) separated by |, for example:
tt01200000|tt012345678|tt0444444
"""
    return prompt

def handle_api_error(error_msg: str, retry_count: int) -> bool:
    """
    Gestisce gli errori dell'API e decide se fare retry.
    
    Args:
        error_msg: messaggio di errore
        retry_count: numero di retry già effettuati
        
    Returns:
        bool: True se dovrebbe fare retry, False altrimenti
    """
    error_msg_lower = error_msg.lower()
    
    # Gestione rate limit
    if "429" in error_msg and "rate limit" in error_msg_lower:
        log(f"⏳ Rate limit raggiunto. Attendo 30 secondi...")
        time.sleep(30)
        return True
    
    # Gestione quota exceeded
    if "quota" in error_msg_lower and "exceeded" in error_msg_lower:
        log(f"⏳ Limite quota raggiunto. Attendo {QUOTA_WAIT_HOURS} ore...")
        time.sleep(QUOTA_WAIT_HOURS * 60 * 60)
        return True
    
    # Gestione altri errori temporanei
    temporary_errors = [
        "timeout", "connection", "network", "service unavailable",
        "temporarily unavailable", "502", "503", "504"
    ]
    
    is_temporary = any(temp_error in error_msg_lower for temp_error in temporary_errors)
    
    if is_temporary and retry_count < MAX_RETRIES:
        log(f"⚠️ Errore temporaneo rilevato (retry {retry_count + 1}/{MAX_RETRIES}): {error_msg}")
        log(f"⏳ Attendo {RETRY_DELAY} secondi prima del retry...")
        time.sleep(RETRY_DELAY)
        return True
    
    return False

def query_together_with_retry(prompt: str, model: str = TOGETHER_MODEL) -> Tuple[str, bool, str, int]:
    """
    Query Together AI API with automatic retry.

    Args:
        prompt: prompt to send
        model: model name

    Returns:
        Tuple[str, bool, str, int]: (response, success, error, retry_count)
    """
    retry_count = 0
    
    while retry_count <= MAX_RETRIES:
        try:
            log(f"📤 Invio query a Together AI {model} (tentativo {retry_count + 1})...")
            
            response = together_client.chat.completions.create(
                temperature=0.1,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a highly skilled movie recommendation system. Always follow instructions carefully."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model=model
            )
            
            log(f"✅ Ricevuta risposta da Together AI")
            return response.choices[0].message.content.strip(), True, "", retry_count

        except Exception as e:
            error_msg = str(e)
            log(f"❌ Errore nella query a Together AI: {error_msg}")
            
            # Check if should retry
            retry_count += 1
            if handle_api_error(error_msg, retry_count - 1) and retry_count <= MAX_RETRIES:
                continue
            else:
                return "", False, error_msg, retry_count
    
    return "", False, f"Massimo numero di retry ({MAX_RETRIES}) raggiunto", MAX_RETRIES

def extract_movie_ids(response: str) -> list:
    """
    Estrae gli ID dei film dalla risposta di Together AI.

    Args:
        response: risposta di Together AI

    Returns:
        list: lista degli ID dei film
    """
    # Cerca pattern di ID film (tt followed by digits)
    tt_pattern = r'tt\d{7,8}'
    movie_ids = re.findall(tt_pattern, response)

    # Se non trova pattern tt, prova a estrarre dalla risposta formattata
    if not movie_ids and '|' in response:
        movie_ids = [id.strip() for id in response.split('|') if id.strip().startswith('tt')]

    # Rimuovi duplicati mantenendo l'ordine
    seen = set()
    unique_ids = []
    for id in movie_ids:
        if id not in seen:
            seen.add(id)
            unique_ids.append(id)

    return unique_ids

def is_negative_query(nl_query: str) -> bool:
    """
    Determina se la query è di tipo negativo (opposto).
    
    Args:
        nl_query: query in linguaggio naturale
        
    Returns:
        bool: True se è una query negativa
    """
    negative_keywords = ['opposto', 'opposite', 'contrario', 'diverso', 'different', 
                        'anti', 'inverse', 'contrary', 'reverse']
    query_lower = nl_query.lower()
    return any(keyword in query_lower for keyword in negative_keywords)

def calculate_ndcg(relevant_scores: List[float], k: int = 10) -> float:
    """
    Calcola NDCG@k.
    
    Args:
        relevant_scores: punteggi di rilevanza per ogni item impostati a 1 se presente nel ground truth, 0 altrimenti
        k: numero di item da considerare
        
    Returns:
        float: valore NDCG@k
    """
    if not relevant_scores or k <= 0:
        return 0.0
    
    # Limita a k elementi
    scores = relevant_scores[:k]
    if not scores:
        return 0.0
    
    # DCG
    dcg = scores[0] + sum(score / np.log2(i + 1) for i, score in enumerate(scores[1:], 2))
    
    # IDCG (Ideal DCG)
    ideal_scores = [k-i for i in range(k)]  # punteggi ideali decrescenti
    idcg = ideal_scores[0] + sum(score / np.log2(i + 1) for i, score in enumerate(ideal_scores[1:], 2))
    
    return dcg / idcg if idcg > 0 else 0.0

def calculate_diversity_score(predicted_ids: List[str], df: pd.DataFrame) -> float:
    """
    Calcola il punteggio di diversità basato sui generi.
    
    Args:
        predicted_ids: lista degli ID predetti
        df: dataframe del dataset
        
    Returns:
        float: punteggio di diversità (0-1)
    """
    if not predicted_ids or 'genre' not in df.columns:
        return 0.0
    
    # Filtra il dataset per gli ID predetti
    predicted_movies = df[df['tconst'].isin(predicted_ids)] if 'tconst' in df.columns else pd.DataFrame()
    
    if predicted_movies.empty:
        return 0.0
    
    # Estrai tutti i generi
    all_genres = set()
    for genres_str in predicted_movies['genre'].dropna():
        if isinstance(genres_str, str):
            genres = [g.strip() for g in genres_str.split(',')]
            all_genres.update(genres)
    
    # Calcola diversità come numero di generi unici / numero totale possibile di generi
    unique_genres = len(all_genres)
    max_possible_genres = min(len(predicted_ids), 20)  # assumendo max 20 generi diversi
    
    return unique_genres / max_possible_genres if max_possible_genres > 0 else 0.0

def calculate_genre_opposition_rate(predicted_ids: List[str], reference_genres: List[str], 
                                  df: pd.DataFrame, k: int = 5) -> float:
    """
    Calcola la percentuale di generi opposti nei top-K.
    
    Args:
        predicted_ids: lista degli ID predetti
        reference_genres: generi di riferimento
        df: dataframe del dataset
        k: numero di top risultati da considerare
        
    Returns:
        float: percentuale di opposizione (0-1)
    """
    if not predicted_ids or not reference_genres or 'genre' not in df.columns:
        return 0.0
    
    # Prendi solo i top-K
    top_k_ids = predicted_ids[:k]
    predicted_movies = df[df['tconst'].isin(top_k_ids)] if 'tconst' in df.columns else pd.DataFrame()
    
    if predicted_movies.empty:
        return 0.0
    
    # Generi opposti (mapping semplificato)
    opposite_genres = {
        'comedy': ['drama', 'horror', 'thriller'],
        'drama': ['comedy', 'action'],
        'action': ['drama', 'romance'],
        'horror': ['comedy', 'romance'],
        'romance': ['horror', 'action', 'thriller'],
        'thriller': ['comedy', 'romance'],
        'adventure': ['drama'],
        'sci-fi': ['historical']
    }
    
    ref_genres_lower = [g.lower().strip() for g in reference_genres]
    opposite_count = 0
    
    for genres_str in predicted_movies['genre'].dropna():
        if isinstance(genres_str, str):
            movie_genres = [g.lower().strip() for g in genres_str.split(',')]
            
            # Controlla se contiene generi opposti
            for ref_genre in ref_genres_lower:
                if ref_genre in opposite_genres:
                    if any(opp_genre in movie_genres for opp_genre in opposite_genres[ref_genre]):
                        opposite_count += 1
                        break
    
    return opposite_count / len(top_k_ids) if top_k_ids else 0.0

def calculate_enhanced_metrics(predicted_ids: List[str], ground_truth_str: str, 
                             nl_query: str, df: pd.DataFrame) -> Dict:
    """
    Calcola tutte le metriche di valutazione, incluse quelle avanzate.
    
    Args:
        predicted_ids: lista degli ID predetti
        ground_truth_str: stringa degli ID ground truth separati da |
        nl_query: query in linguaggio naturale originale
        df: dataframe del dataset
        
    Returns:
        dict: tutte le metriche calcolate
    """
    # Metriche base
    basic_metrics = calculate_metrics(predicted_ids, ground_truth_str)
    
    # Determina se è una query negativa
    is_negative = is_negative_query(nl_query)
    
    # Inizializza metriche avanzate
    advanced_metrics = {
        'is_negative_query': is_negative,
        'ndcg_10': 0.0,
        'precision_5': 0.0,
        'precision_10': 0.0,
        'spearman_correlation': 0.0,
        'diversity_score': 0.0,
        'genre_opposition_rate_5': 0.0,
        'genre_opposition_rate_10': 0.0
    }
    
    if not ground_truth_str or pd.isna(ground_truth_str) or not predicted_ids:
        return {**basic_metrics, **advanced_metrics}
    
    ground_truth_ids = str(ground_truth_str).split('|')
    
    # Calcola punteggi di rilevanza (1 se è nel ground truth, 0 altrimenti)
    k = len(ground_truth_ids)
    relevance_scores = [k - i if pid in ground_truth_ids else 0 for i, pid in enumerate(predicted_ids)]
    
    # NDCG@10
    advanced_metrics['ndcg_10'] = calculate_ndcg(relevance_scores, 10)
    
    # Precision@K
    if len(predicted_ids) >= 5:
        top_5_relevant = sum(1 for pid in predicted_ids[:5] if pid in ground_truth_ids)
        advanced_metrics['precision_5'] = top_5_relevant / 5
    
    if len(predicted_ids) >= 10:
        top_10_relevant = sum(1 for pid in predicted_ids[:10] if pid in ground_truth_ids)
        advanced_metrics['precision_10'] = top_10_relevant / 10
    
    # Spearman correlation (se abbiamo abbastanza dati)
    if len(predicted_ids) >= 3 and len(ground_truth_ids) >= 3:
        try:
            # Crea ranking per predicted e ground truth
            pred_ranks = {pid: i for i, pid in enumerate(predicted_ids)}
            gt_ranks = {gid: i for i, gid in enumerate(ground_truth_ids)}
            
            # Trova intersezione
            common_ids = set(predicted_ids) & set(ground_truth_ids)
            if len(common_ids) >= 3:
                pred_common_ranks = [pred_ranks[cid] for cid in common_ids]
                gt_common_ranks = [gt_ranks[cid] for cid in common_ids]
                
                correlation, _ = spearmanr(pred_common_ranks, gt_common_ranks)
                advanced_metrics['spearman_correlation'] = correlation if not np.isnan(correlation) else 0.0
        except:
            pass
    
    # Diversity Score
    advanced_metrics['diversity_score'] = calculate_diversity_score(predicted_ids, df)
    
    # Genre Opposition Rate (solo per query negative)
    if is_negative and 'genre' in df.columns:
        # Cerca di estrarre generi di riferimento dalla query o dal ground truth
        reference_genres = []
        if ground_truth_ids and 'tconst' in df.columns:
            ref_movies = df[df['tconst'].isin(ground_truth_ids[:3])]  # usa i primi 3 del ground truth
            for genres_str in ref_movies['genre'].dropna():
                if isinstance(genres_str, str):
                    reference_genres.extend([g.strip() for g in genres_str.split(',')])
        
        if reference_genres:
            advanced_metrics['genre_opposition_rate_5'] = calculate_genre_opposition_rate(
                predicted_ids, reference_genres, df, 5)
            advanced_metrics['genre_opposition_rate_10'] = calculate_genre_opposition_rate(
                predicted_ids, reference_genres, df, 10)
    
    return {**basic_metrics, **advanced_metrics}

def calculate_metrics(predicted_ids: list, ground_truth_str: str) -> dict:
    """
    Calcola le metriche di valutazione base.

    Args:
        predicted_ids: lista degli ID predetti
        ground_truth_str: stringa degli ID ground truth separati da |

    Returns:
        dict: metriche calcolate
    """
    if not ground_truth_str or pd.isna(ground_truth_str):
        return {
            'precision': 0, 
            'recall': 0, 
            'f1': 0, 
            'accuracy': 0,
            'intersection_size': 0,
            'ground_truth_size': 0,
            'predicted_size': len(predicted_ids)
        }

    ground_truth_ids = set(str(ground_truth_str).split('|'))
    predicted_set = set(predicted_ids)

    # Calcola intersezione
    intersection = ground_truth_ids.intersection(predicted_set)

    # Calcola metriche
    precision = len(intersection) / len(predicted_set) if predicted_set else 0
    recall = len(intersection) / len(ground_truth_ids) if ground_truth_ids else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = len(intersection) / len(ground_truth_ids.union(predicted_set)) if ground_truth_ids.union(predicted_set) else 0

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
        'intersection_size': len(intersection),
        'ground_truth_size': len(ground_truth_ids),
        'predicted_size': len(predicted_set)
    }

def load_progress(output_path: str):
    """
    Carica il progresso salvato dal file di output.
    
    Args:
        output_path: path del file di output
    
    Returns:
        set: insieme degli ID già processati
    """
    if os.path.exists(output_path):
        try:
            existing_df = pd.read_csv(output_path)
            # Restituisce gli ID già processati con successo
            successful_ids = existing_df[existing_df['success'] == True]['test_id'].tolist()
            return set(successful_ids)
        except Exception as e:
            log(f"⚠️ Errore nel caricamento del progresso: {e}")
            return set()
    return set()

def save_result_incremental(result_row: dict, output_path: str, test_df_columns: list):
    """
    Salva incrementalmente un risultato nel file di output.
    
    Args:
        result_row: dizionario con il risultato del test
        output_path: path del file di output
        test_df_columns: colonne originali del test dataframe
    """
    # Prepara il dataframe per il salvataggio
    df_to_save = pd.DataFrame([result_row])
    
    # Riordina le colonne per avere prima quelle originali, poi quelle aggiunte
    original_cols = [col for col in test_df_columns if col in df_to_save.columns]
    new_cols = [col for col in df_to_save.columns if col not in test_df_columns]
    ordered_cols = original_cols + new_cols
    df_to_save = df_to_save[ordered_cols]
    
    # Se il file non esiste, crea con header
    if not os.path.exists(output_path):
        df_to_save.to_csv(output_path, index=False, mode='w')
        log(f"📝 Creato nuovo file di risultati: {output_path}")
    else:
        # Altrimenti appendi senza header
        df_to_save.to_csv(output_path, index=False, mode='a', header=False)
    
    log(f"💾 Salvato risultato test_id={result_row['test_id']} nel file")

def process_test_row(row: dict, exec_type: str = "NORMAL", model: str = TOGETHER_MODEL) -> dict:
    """
    Processa una singola riga di test con retry automatico.

    Args:
        row: riga del test (dizionario con tutte le colonne originali)
        exec_type: modalità di esecuzione
        model: modello Together AI da utilizzare

    Returns:
        dict: risultato del test con tutte le colonne originali + metriche
    """
    log(f"➡️ Processando test: {row['nl_query'][:50]}... (modalità: {exec_type}, modello: {model})")

    # Inizializza il risultato con tutte le colonne originali
    result = row.copy()
    
    # Aggiungi le nuove colonne per i risultati
    result['exec_type'] = exec_type
    result['model'] = model
    result['processing_timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Carica il dataset
    df = load_dataset(row['csv_path'], exec_type)
    if df is None:
        result.update({
            'success': False,
            'error': f"Impossibile caricare dataset da {row['csv_path']}",
            'predicted_ids': [],
            'predicted_ids_str': '',
            'raw_response': '',
            'dataset_shape': '',
            'retry_count': 0,
            # Metriche base
            'precision': 0, 'recall': 0, 'f1': 0, 'accuracy': 0,
            'intersection_size': 0, 'ground_truth_size': 0, 'predicted_size': 0,
            # Metriche avanzate
            'is_negative_query': False, 'ndcg_10': 0.0, 'precision_5': 0.0, 'precision_10': 0.0,
            'spearman_correlation': 0.0, 'diversity_score': 0.0,
            'genre_opposition_rate_5': 0.0, 'genre_opposition_rate_10': 0.0
        })
        return result

    # Formatta il dataset per il prompt
    dataset_str = format_dataset_for_prompt(df)

    # Crea il prompt
    prompt = create_prompt(dataset_str, row['nl_query'])

    # Query Together AI con retry
    response, success, error_msg, retry_count = query_together_with_retry(prompt, model)
    
    if not success:
        result.update({
            'success': False,
            'error': error_msg,
            'predicted_ids': [],
            'predicted_ids_str': '',
            'raw_response': '',
            'dataset_shape': f"{df.shape[0]}x{df.shape[1]}",
            'retry_count': retry_count,
            # Metriche base
            'precision': 0, 'recall': 0, 'f1': 0, 'accuracy': 0,
            'intersection_size': 0, 'ground_truth_size': 0, 'predicted_size': 0,
            # Metriche avanzate
            'is_negative_query': False, 'ndcg_10': 0.0, 'precision_5': 0.0, 'precision_10': 0.0,
            'spearman_correlation': 0.0, 'diversity_score': 0.0,
            'genre_opposition_rate_5': 0.0, 'genre_opposition_rate_10': 0.0
        })
        return result

    # Estrai gli ID dei film
    predicted_ids = extract_movie_ids(response)

    # Calcola tutte le metriche (base + avanzate)
    all_metrics = calculate_enhanced_metrics(predicted_ids, row.get('ground_truth', ''), 
                                           row['nl_query'], df)

    # Aggiorna il risultato con tutti i dati
    result.update({
        'success': True,
        'error': '',
        'predicted_ids': predicted_ids,
        'predicted_ids_str': '|'.join(predicted_ids),
        'raw_response': response[:500] if len(response) > 500 else response,
        'dataset_shape': f"{df.shape[0]}x{df.shape[1]}",
        'retry_count': retry_count,
        **all_metrics  # Aggiunge tutte le metriche (base + avanzate)
    })

    return result

def run_tests_from_csv(test_csv_path: str, output_dir: str = "results", 
                       exec_types: list = ["NORMAL"], model: str = TOGETHER_MODEL):
    """
    Esegue i test da un file CSV con salvataggio incrementale e ripresa del progresso.

    Args:
        test_csv_path: path al file CSV dei test
        output_dir: directory di output per i risultati
        exec_types: lista delle modalità di esecuzione da testare
        model: modello Together AI da utilizzare
    """
    # Assicurati che la directory di output esista
    os.makedirs(output_dir, exist_ok=True)

    # Carica i test
    try:
        test_df = pd.read_csv(test_csv_path)
        log(f"📊 Caricati {len(test_df)} test da {test_csv_path}")
        
        # Aggiungi test_id se non presente
        if 'test_id' not in test_df.columns:
            test_df['test_id'] = range(len(test_df))
        
        test_columns = test_df.columns.tolist()
        
    except Exception as e:
        log(f"❌ Errore nel caricamento dei test: {e}")
        return

    # Esegui per ogni modalità
    for exec_type in exec_types:
        log(f"🚀 Avvio test in modalità {exec_type} con modello {model}")
        
        # Definisci il path di output per questa modalità
        model_safe = model.replace('/', '_').replace('-', '_')
        output_path = os.path.join(output_dir, f"results_{exec_type.lower()}_{model_safe}.csv")
        
        # Carica il progresso
        processed_ids = load_progress(output_path)
        if processed_ids:
            log(f"📈 Trovati {len(processed_ids)} test già completati con successo, riprendo dal test successivo")
        
        # Contatori per le statistiche
        total_processed = 0
        total_success = 0
        cumulative_metrics = {
            'f1': [], 'precision': [], 'recall': [], 'accuracy': [],
            'ndcg_10': [], 'diversity_score': [], 'spearman_correlation': []
        }
        
        for idx, row in test_df.iterrows():
            test_id = row.get('test_id', idx)
            
            # Salta se già processato con successo
            if test_id in processed_ids:
                log(f"⏭️ Test {test_id} già processato con successo, salto al successivo")
                continue
            
            try:
                # Processa il test
                result = process_test_row(row.to_dict(), exec_type, model)
                result['test_id'] = test_id
                
                # Salva immediatamente il risultato
                save_result_incremental(result, output_path, test_columns)
                
                # Aggiorna le statistiche
                total_processed += 1
                if result['success']:
                    total_success += 1
                    # Metriche base
                    cumulative_metrics['f1'].append(result.get('f1', 0))
                    cumulative_metrics['precision'].append(result.get('precision', 0))
                    cumulative_metrics['recall'].append(result.get('recall', 0))
                    cumulative_metrics['accuracy'].append(result.get('accuracy', 0))
                    # Metriche avanzate
                    cumulative_metrics['ndcg_10'].append(result.get('ndcg_10', 0))
                    cumulative_metrics['diversity_score'].append(result.get('diversity_score', 0))
                    cumulative_metrics['spearman_correlation'].append(result.get('spearman_correlation', 0))
                    
                    log(f"✅ Test {test_id} completato - F1: {result.get('f1', 0):.3f}, "
                        f"Precision: {result.get('precision', 0):.3f}, "
                        f"Recall: {result.get('recall', 0):.3f}, "
                        f"NDCG@10: {result.get('ndcg_10', 0):.3f}, "
                        f"Diversity: {result.get('diversity_score', 0):.3f}, "
                        f"Predetti: {result.get('predicted_size', 0)} film")
                else:
                    log(f"❌ Test {test_id} fallito: {result.get('error', 'Errore sconosciuto')}")
                
                # Stampa progresso ogni 10 test
                if total_processed % 10 == 0:
                    log(f"📊 Progresso: {total_processed} test processati, "
                        f"{len(processed_ids) + total_processed}/{len(test_df)} totali")
                
                # Pausa per evitare rate limiting
                time.sleep(3)

            except KeyboardInterrupt:
                log(f"⚠️ Interruzione manuale al test {test_id}")
                break
                
            except Exception as e:
                log(f"❌ Errore nel test {test_id}: {e}")
                error_result = row.to_dict()
                error_result.update({
                    'test_id': test_id,
                    'success': False,
                    'error': str(e),
                    'exec_type': exec_type,
                    'model': model,
                    'predicted_ids': [],
                    'predicted_ids_str': '',
                    'processing_timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'retry_count': MAX_RETRIES,
                    # Metriche base
                    'precision': 0, 'recall': 0, 'f1': 0, 'accuracy': 0,
                    'intersection_size': 0, 'ground_truth_size': 0, 'predicted_size': 0,
                    # Metriche avanzate
                    'is_negative_query': False, 'ndcg_10': 0.0, 'precision_5': 0.0, 'precision_10': 0.0,
                    'spearman_correlation': 0.0, 'diversity_score': 0.0,
                    'genre_opposition_rate_5': 0.0, 'genre_opposition_rate_10': 0.0
                })
                save_result_incremental(error_result, output_path, test_columns)

        # Stampa statistiche finali per questa modalità
        log(f"🎯 Completati test per modalità {exec_type}")
        log(f"📊 Test processati in questa sessione: {total_processed}")
        log(f"✅ Test riusciti: {total_success}/{total_processed}")
        
        if cumulative_metrics['f1']:
            # Metriche base
            avg_f1 = sum(cumulative_metrics['f1']) / len(cumulative_metrics['f1'])
            avg_precision = sum(cumulative_metrics['precision']) / len(cumulative_metrics['precision'])
            avg_recall = sum(cumulative_metrics['recall']) / len(cumulative_metrics['recall'])
            avg_accuracy = sum(cumulative_metrics['accuracy']) / len(cumulative_metrics['accuracy'])
            
            # Metriche avanzate
            avg_ndcg = sum(cumulative_metrics['ndcg_10']) / len(cumulative_metrics['ndcg_10'])
            avg_diversity = sum(cumulative_metrics['diversity_score']) / len(cumulative_metrics['diversity_score'])
            avg_spearman = sum(cumulative_metrics['spearman_correlation']) / len(cumulative_metrics['spearman_correlation'])
            
            log(f"📈 Metriche medie {exec_type}:")
            log(f"   📊 Metriche Base:")
            log(f"      • F1: {avg_f1:.3f}")
            log(f"      • Precision: {avg_precision:.3f}")
            log(f"      • Recall: {avg_recall:.3f}")
            log(f"      • Accuracy: {avg_accuracy:.3f}")
            log(f"   🎯 Metriche Avanzate:")
            log(f"      • NDCG@10: {avg_ndcg:.3f}")
            log(f"      • Diversity Score: {avg_diversity:.3f}")
            log(f"      • Spearman Correlation: {avg_spearman:.3f}")

def run_single_test(csv_path: str, nl_query: str, exec_type: str = "NORMAL", 
                    model: str = TOGETHER_MODEL) -> dict:
    """
    Esegue un singolo test.

    Args:
        csv_path: path al dataset CSV
        nl_query: query in linguaggio naturale
        exec_type: modalità di esecuzione
        model: modello Together AI da utilizzare

    Returns:
        dict: risultato del test
    """
    test_row = {
        'csv_path': csv_path,
        'nl_query': nl_query,
        'test_id': 0
    }

    return process_test_row(test_row, exec_type, model)

# === FUNZIONI DI UTILITÀ PER ANALISI RISULTATI ===

def analyze_results(results_csv_path: str) -> Dict:
    """
    Analizza i risultati dei test e genera statistiche complete.
    
    Args:
        results_csv_path: path al file CSV dei risultati
        
    Returns:
        dict: statistiche complete dei risultati
    """
    try:
        df = pd.read_csv(results_csv_path)
        log(f"📊 Analizzando {len(df)} risultati da {results_csv_path}")
        
        # Statistiche generali
        total_tests = len(df)
        successful_tests = len(df[df['success'] == True])
        failed_tests = total_tests - successful_tests
        success_rate = successful_tests / total_tests if total_tests > 0 else 0
        
        # Statistiche per query positive e negative
        positive_queries = df[df['is_negative_query'] == False]
        negative_queries = df[df['is_negative_query'] == True]
        
        stats = {
            'general': {
                'total_tests': total_tests,
                'successful_tests': successful_tests,
                'failed_tests': failed_tests,
                'success_rate': success_rate,
                'positive_queries': len(positive_queries),
                'negative_queries': len(negative_queries)
            }
        }
        
        # Metriche per query positive
        if not positive_queries.empty:
            pos_successful = positive_queries[positive_queries['success'] == True]
            if not pos_successful.empty:
                stats['positive_queries'] = {
                    'count': len(pos_successful),
                    'avg_f1': pos_successful['f1'].mean(),
                    'avg_precision': pos_successful['precision'].mean(),
                    'avg_recall': pos_successful['recall'].mean(),
                    'avg_ndcg_10': pos_successful['ndcg_10'].mean(),
                    'avg_precision_5': pos_successful['precision_5'].mean(),
                    'avg_precision_10': pos_successful['precision_10'].mean(),
                    'avg_spearman': pos_successful['spearman_correlation'].mean(),
                    'avg_diversity': pos_successful['diversity_score'].mean()
                }
        
        # Metriche per query negative
        if not negative_queries.empty:
            neg_successful = negative_queries[negative_queries['success'] == True]
            if not neg_successful.empty:
                stats['negative_queries'] = {
                    'count': len(neg_successful),
                    'avg_f1': neg_successful['f1'].mean(),
                    'avg_precision': neg_successful['precision'].mean(),
                    'avg_recall': neg_successful['recall'].mean(),
                    'avg_ndcg_10': neg_successful['ndcg_10'].mean(),
                    'avg_diversity': neg_successful['diversity_score'].mean(),
                    'avg_genre_opposition_5': neg_successful['genre_opposition_rate_5'].mean(),
                    'avg_genre_opposition_10': neg_successful['genre_opposition_rate_10'].mean()
                }
        
        return stats
        
    except Exception as e:
        log(f"❌ Errore nell'analisi dei risultati: {e}")
        return {}

def print_analysis_report(stats: Dict):
    """
    Stampa un report dettagliato delle statistiche.
    
    Args:
        stats: dizionario delle statistiche
    """
    if not stats:
        log("❌ Nessuna statistica disponibile")
        return
    
    log("\n" + "="*60)
    log("📊 REPORT ANALISI RISULTATI")
    log("="*60)
    
    # Statistiche generali
    general = stats.get('general', {})
    log(f"\n🎯 STATISTICHE GENERALI:")
    log(f"   • Test totali: {general.get('total_tests', 0)}")
    log(f"   • Test riusciti: {general.get('successful_tests', 0)}")
    log(f"   • Test falliti: {general.get('failed_tests', 0)}")
    log(f"   • Tasso di successo: {general.get('success_rate', 0):.1%}")
    log(f"   • Query positive: {general.get('positive_queries', 0)}")
    log(f"   • Query negative: {general.get('negative_queries', 0)}")
    
    # Statistiche query positive
    if 'positive_queries' in stats:
        pos = stats['positive_queries']
        log(f"\n✅ QUERY POSITIVE ({pos.get('count', 0)} test):")
        log(f"   📊 Metriche Base:")
        log(f"      • F1 Score: {pos.get('avg_f1', 0):.3f}")
        log(f"      • Precision: {pos.get('avg_precision', 0):.3f}")
        log(f"      • Recall: {pos.get('avg_recall', 0):.3f}")
        log(f"   🎯 Metriche Avanzate:")
        log(f"      • NDCG@10: {pos.get('avg_ndcg_10', 0):.3f}")
        log(f"      • Precision@5: {pos.get('avg_precision_5', 0):.3f}")
        log(f"      • Precision@10: {pos.get('avg_precision_10', 0):.3f}")
        log(f"      • Spearman Correlation: {pos.get('avg_spearman', 0):.3f}")
        log(f"      • Diversity Score: {pos.get('avg_diversity', 0):.3f}")
    
    # Statistiche query negative
    if 'negative_queries' in stats:
        neg = stats['negative_queries']
        log(f"\n❌ QUERY NEGATIVE ({neg.get('count', 0)} test):")
        log(f"   📊 Metriche Base:")
        log(f"      • F1 Score: {neg.get('avg_f1', 0):.3f}")
        log(f"      • Precision: {neg.get('avg_precision', 0):.3f}")
        log(f"      • Recall: {neg.get('avg_recall', 0):.3f}")
        log(f"   🎯 Metriche Specializzate:")
        log(f"      • NDCG@10 Inverso: {neg.get('avg_ndcg_10', 0):.3f}")
        log(f"      • Diversity Score: {neg.get('avg_diversity', 0):.3f}")
        log(f"      • Genre Opposition@5: {neg.get('avg_genre_opposition_5', 0):.3f}")
        log(f"      • Genre Opposition@10: {neg.get('avg_genre_opposition_10', 0):.3f}")
    
    log("\n" + "="*60)

# === MAIN ENTRY POINT ===
if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description='Bot avanzato per testare le API di Together AI con dataset cinematografici')
    parser.add_argument('--test_csv', required=True, help='Path al file CSV dei test')
    parser.add_argument('--output_dir', default='results', help='Directory di output per i risultati')
    parser.add_argument('--modes', nargs='+', default=['NORMAL'], choices=['NORMAL', 'NULL', 'REMOVE'], 
                       help='Modalità di esecuzione da testare')
    parser.add_argument('--rem_columns', nargs='+', default=['title', 'year'],
                       help='Colonne da rimuovere/nullificare')
    parser.add_argument('--model', default=TOGETHER_MODEL, help='Nome del modello Together AI da utilizzare')
    parser.add_argument('--analyze', action='store_true', 
                       help='Analizza i risultati esistenti invece di eseguire nuovi test')
    parser.add_argument('--results_file', help='Path al file dei risultati da analizzare (usato con --analyze)')

    args = parser.parse_args()

    # Aggiorna la variabile globale delle colonne
    rem_columns = args.rem_columns

    # Imposta il modello di Together AI
    TOGETHER_MODEL_TO_USE = args.model
    log(f"🤖 Modello Together AI in uso: {TOGETHER_MODEL_TO_USE}")

    if args.analyze:
        # Modalità analisi
        if not args.results_file:
            log("❌ Specificare --results_file quando si usa --analyze")
            sys.exit(1)
        
        log(f"📊 Avvio analisi risultati da {args.results_file}")
        stats = analyze_results(args.results_file)
        print_analysis_report(stats)
    else:
        # Modalità test normale
        log(f"🤖 Avvio bot per test Together AI con retry automatico")
        log(f"📁 File test: {args.test_csv}")
        log(f"📂 Output: {args.output_dir}")
        log(f"⚙️ Modalità: {args.modes}")
        log(f"📋 Colonne da modificare: {rem_columns}")
        log(f"🔄 Max retry per test: {MAX_RETRIES}")
        log(f"⏱️ Attesa tra retry: {RETRY_DELAY}s")
        log(f"⏳ Attesa per quota exceeded: {QUOTA_WAIT_HOURS}h")

        # Esegui i test
        run_tests_from_csv(args.test_csv, args.output_dir, args.modes, TOGETHER_MODEL_TO_USE)

        log(f"🎯 Test completati!")