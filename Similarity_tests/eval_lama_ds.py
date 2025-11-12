import pandas as pd
import numpy as np
import re
import os
from pathlib import Path
from typing import List, Dict
from scipy.stats import spearmanr
from datetime import datetime

# ============== FUNZIONI DI ESTRAZIONE E UTILITÀ ==============

def extract_movie_ids(response: str) -> list:
    """
    Estrae gli ID dei film dalla risposta.
    """
    tt_pattern = r'tt\d{7,8}'
    movie_ids = re.findall(tt_pattern, response)

    if not movie_ids and '|' in response:
        movie_ids = [id.strip() for id in response.split('|') if id.strip().startswith('tt')]

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
    """
    negative_keywords = ['opposto', 'opposite', 'contrario', 'diverso', 'different', 
                        'anti', 'inverse', 'contrary', 'reverse']
    query_lower = nl_query.lower()
    return any(keyword in query_lower for keyword in negative_keywords)

# ============== FUNZIONI DI CALCOLO METRICHE ==============

def calculate_ndcg(relevant_scores: List[float], k: int = 10) -> float:
    """
    Calcola NDCG@k.
    
    Args:
        relevant_scores: punteggi di rilevanza per ogni item
        k: numero di item da considerare
        
    Returns:
        float: valore NDCG@k
    """
    """Calcola NDCG@k."""
    if not relevant_scores or k <= 0:
        return 0.0

    scores = relevant_scores[:k]
    if not scores:
        return 0.0

    dcg = scores[0] + sum(score / np.log2(i + 1) for i, score in enumerate(scores[1:], 2))
    ideal_scores = [k - i for i in range(k)]
    idcg = ideal_scores[0] + sum(score / np.log2(i + 1) for i, score in enumerate(ideal_scores[1:], 2))

    return dcg / idcg if idcg > 0 else 0.0

def calculate_diversity_score(predicted_ids: List[str], df: pd.DataFrame) -> float:
    """
    Calcola il punteggio di diversità basato sui generi.
    """
    if not predicted_ids or 'genre' not in df.columns:
        return 0.0
    
    predicted_movies = df[df['tconst'].isin(predicted_ids)] if 'tconst' in df.columns else pd.DataFrame()
    
    if predicted_movies.empty:
        return 0.0
    
    all_genres = set()
    for genres_str in predicted_movies['genre'].dropna():
        if isinstance(genres_str, str):
            genres = [g.strip() for g in genres_str.split(',')]
            all_genres.update(genres)
    
    unique_genres = len(all_genres)
    max_possible_genres = min(len(predicted_ids), 20)
    
    return unique_genres / max_possible_genres if max_possible_genres > 0 else 0.0

def calculate_genre_opposition_rate(predicted_ids: List[str], reference_genres: List[str], 
                                  df: pd.DataFrame, k: int = 5) -> float:
    """
    Calcola la percentuale di generi opposti nei top-K.
    """
    if not predicted_ids or not reference_genres or 'genre' not in df.columns:
        return 0.0
    
    top_k_ids = predicted_ids[:k]
    predicted_movies = df[df['tconst'].isin(top_k_ids)] if 'tconst' in df.columns else pd.DataFrame()
    
    if predicted_movies.empty:
        return 0.0
    
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
            
            for ref_genre in ref_genres_lower:
                if ref_genre in opposite_genres:
                    if any(opp_genre in movie_genres for opp_genre in opposite_genres[ref_genre]):
                        opposite_count += 1
                        break
    
    return opposite_count / len(top_k_ids) if top_k_ids else 0.0

def calculate_metrics(predicted_ids: list, ground_truth_str: str) -> dict:
    """
    Calcola le metriche di valutazione base.
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

    intersection = ground_truth_ids.intersection(predicted_set)

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

def calculate_enhanced_metrics(predicted_ids: List[str], ground_truth_str: str, 
                             nl_query: str, df: pd.DataFrame) -> Dict:
    """
    Calcola tutte le metriche di valutazione, incluse quelle avanzate.
    """
    basic_metrics = calculate_metrics(predicted_ids, ground_truth_str)
    
    is_negative = is_negative_query(nl_query)
    
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
    
    k = len(ground_truth_ids)
    relevance_scores = [k - i if pid in ground_truth_ids else 0 for i, pid in enumerate(predicted_ids)]
    
    advanced_metrics['ndcg_10'] = calculate_ndcg(relevance_scores, k)
    
    if len(predicted_ids) >= 5:
        top_5_relevant = sum(1 for pid in predicted_ids[:5] if pid in ground_truth_ids)
        advanced_metrics['precision_5'] = top_5_relevant / 5
    
    if len(predicted_ids) >= 10:
        top_10_relevant = sum(1 for pid in predicted_ids[:10] if pid in ground_truth_ids)
        advanced_metrics['precision_10'] = top_10_relevant / 10
    
    if len(predicted_ids) >= 3 and len(ground_truth_ids) >= 3:
        try:
            pred_ranks = {pid: i for i, pid in enumerate(predicted_ids)}
            gt_ranks = {gid: i for i, gid in enumerate(ground_truth_ids)}
            
            common_ids = set(predicted_ids) & set(ground_truth_ids)
            if len(common_ids) >= 3:
                pred_common_ranks = [pred_ranks[cid] for cid in common_ids]
                gt_common_ranks = [gt_ranks[cid] for cid in common_ids]
                
                correlation, _ = spearmanr(pred_common_ranks, gt_common_ranks)
                advanced_metrics['spearman_correlation'] = correlation if not np.isnan(correlation) else 0.0
        except:
            pass
    
    advanced_metrics['diversity_score'] = calculate_diversity_score(predicted_ids, df)
    
    if is_negative and 'genre' in df.columns:
        reference_genres = []
        if ground_truth_ids and 'tconst' in df.columns:
            ref_movies = df[df['tconst'].isin(ground_truth_ids[:3])]
            for genres_str in ref_movies['genre'].dropna():
                if isinstance(genres_str, str):
                    reference_genres.extend([g.strip() for g in genres_str.split(',')])
        
        if reference_genres:
            advanced_metrics['genre_opposition_rate_5'] = calculate_genre_opposition_rate(
                predicted_ids, reference_genres, df, 5)
            advanced_metrics['genre_opposition_rate_10'] = calculate_genre_opposition_rate(
                predicted_ids, reference_genres, df, 10)
    
    return {**basic_metrics, **advanced_metrics}

# ============== FUNZIONI PRINCIPALI ==============

def load_dataset(csv_path: str) -> pd.DataFrame:
    """
    Carica il dataset IMDB se esiste.
    """
    if os.path.exists(csv_path):
        try:
            return pd.read_csv(csv_path)
        except Exception as e:
            print(f"Errore nel caricamento del dataset {csv_path}: {e}")
    return pd.DataFrame()

def process_single_csv(file_path: str, output_dir: str = None) -> pd.DataFrame:
    """
    Processa un singolo file CSV e calcola le metriche.
    """
    print(f"\nProcessing: {file_path}")
    
    try:
        df = pd.read_csv(file_path)
        
        # Verifica colonne richieste
        required_cols = ['ground_truth', 'predicted_ids_str', 'nl_query', 'csv_path']
        if not all(col in df.columns for col in required_cols):
            print(f"  ⚠ Colonne mancanti in {file_path}")
            return None
        
        # Carica dataset IMDB (prendi il primo csv_path come riferimento)
        dataset_path = df['csv_path'].iloc[0] if len(df) > 0 else None
        imdb_df = load_dataset(dataset_path) if dataset_path else pd.DataFrame()
        
        # Calcola metriche per ogni riga
        results = []
        for idx, row in df.iterrows():
            try:
                # Estrai predicted_ids da predicted_ids_str
                predicted_ids = extract_movie_ids(str(row['predicted_ids_str']))
                
                # Calcola metriche
                metrics = calculate_enhanced_metrics(
                    predicted_ids,
                    str(row['ground_truth']),
                    str(row['nl_query']),
                    imdb_df
                )
                
                # Aggiungi dati originali
                result = {**row.to_dict(), **metrics}
                results.append(result)
                
            except Exception as e:
                print(f"  ⚠ Errore nella riga {idx}: {e}")
                continue
        
        result_df = pd.DataFrame(results)
        
        # Salva risultati
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(
                output_dir, 
                f"{Path(file_path).stem}_metrics.csv"
            )
            result_df.to_csv(output_file, index=False)
            print(f"  ✓ Salvato: {output_file}")
        
        return result_df
        
    except Exception as e:
        print(f"  ✗ Errore: {e}")
        return None

def process_folder(folder_path: str, output_dir: str = "metrics_output"):
    """
    Processa tutti i CSV in una cartella e sottocartelle.
    """
    print(f"Cercando file CSV in: {folder_path}")
    
    csv_files = list(Path(folder_path).rglob("*.csv"))
    
    if not csv_files:
        print("Nessun file CSV trovato!")
        return
    
    print(f"Trovati {len(csv_files)} file CSV\n")
    
    processed_count = 0
    
    for csv_file in csv_files:
        result_df = process_single_csv(str(csv_file), output_dir)
        if result_df is not None:
            processed_count += 1
    
    print(f"\n{'='*60}")
    print(f"✓ Completato!")
    print(f"✓ File processati: {processed_count}/{len(csv_files)}")
    print(f"✓ Risultati salvati in: {output_dir}")
    print(f"{'='*60}")

# ============== MAIN ==============

if __name__ == "__main__":
    # Configurazione
    INPUT_FOLDER = "results_copy/Lama_DS/cut_300"  # Cartella con i CSV da processare
    OUTPUT_FOLDER = "results_copy/Lama_DS_eval/cut_300"  # Cartella per i risultati
    
    # Processa tutti i file
    process_folder(INPUT_FOLDER, OUTPUT_FOLDER)