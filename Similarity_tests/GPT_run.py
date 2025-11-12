import pandas as pd
import sqlite3
import re
import json
from playwright.sync_api import sync_playwright
import time
import os
import random
import requests
import gc
import sqlparse
from sqlparse.sql import IdentifierList, Identifier
from sqlparse.tokens import Keyword, DML
import sys
from dotenv import load_dotenv
from datetime import datetime
import traceback
import numpy as np
from typing import List, Dict, Tuple, Optional
from scipy.stats import spearmanr




# === CONFIG ===
model = 'gpt-5_mini'
progress_path = 'progress_gpt.json'
tests_json = 'tests_GPT.json'
chrome_path = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
chrome_profile = "/Users/seba/Library/Application Support/Google/Chrome/Profile 5"
rem_col = ['IMDB_id']
#['year','genre','directors','writers','main_cast','duration_min','AVG_score','number_of_votes']

RESTART_BROWSER_EVERY = 50    # richieste prima di riavvio browser
RESET_CHAT_EVERY = 25         # richieste prima di nuova chat
WAIT_AFTER_REPLY = (20, 25)    # attesa random dopo risposta in secondi
MAX_RETRIES = 3               # maximum number of retry attempts
# =================

with open(tests_json, 'r') as f:
    tests_list = json.load(f)

if os.path.exists(progress_path):
    with open(progress_path, 'r') as f:
        progress = json.load(f)
else:
    progress = {}

def launch_browser(p):
    browser = p.chromium.launch_persistent_context(
        user_data_dir=chrome_profile,
        executable_path=chrome_path,
        headless=False,
        args=["--disable-blink-features=AutomationControlled", "--start-maximized", "--no-activate-on-launch"]
    )
    page = browser.new_page()
    page.goto("https://chat.openai.com")
    page.wait_for_selector('div.ProseMirror[contenteditable="true"]', timeout=60000)
    return browser, page

def send_system_message(page):
    page.wait_for_selector('div.ProseMirror[contenteditable="true"]', timeout=60000)
    input_box = page.locator('div.ProseMirror[contenteditable="true"]')
    input_box.click()
    page.wait_for_timeout(1000)
    system_msg = """You are a movie recommendation system. Analyze the provided movie dataset and respond to the user's request. allways respond in valid json inside code blocks."""
    input_box.fill(system_msg)
    time.sleep(1)
    input_box.press("Enter")
    page.wait_for_timeout(3000)

def load_tables_for_query(csv_path, exec_type):
    df = pd.read_csv(csv_path, dtype=str)

    if exec_type == "NULL":
        for col in rem_col:
            df[col] = df[col].apply(lambda _: None)

    elif exec_type == "REMOVE":
        df = df.drop(columns=[col for col in rem_col if col in df.columns], errors='ignore')

    return df.to_json()
import re
import json

def extract_json_from_page(page):
    """
    Estrae e corregge JSON dalla pagina, gestendo anche formati non standard.
    """
    candidate_selectors = [
        "div[data-message-author-role='assistant'] code.language-json",
        "div[data-message-author-role='assistant'] code",
        "div[data-message-author-role='assistant'] pre code",
        "div[data-message-author-role='assistant'] pre",
    ]
    
    json_candidates = []
    
    for selector in candidate_selectors:
        elements = page.locator(selector)
        for i in range(elements.count()):
            text = elements.nth(i).evaluate("el => el.textContent")
            
            if not text or "{" not in text or "}" not in text:
                continue
            
            try:
                start = text.index("{")
                end = text.rindex("}") + 1
                json_str = text[start:end]
                
                # Pulizia caratteri invisibili
                json_str = json_str.replace('\u200b', '')
                json_str = json_str.replace('\xa0', ' ')
                
                # FIX: Correggi il formato { "tt123" } -> "tt123"
                # Pattern: trova { "ttXXXXXX" } e sostituisci con "ttXXXXXX"
                json_str = re.sub(r'\{\s*"(tt\d+)"\s*\}', r'"\1"', json_str)
                
                # Prova a parsare
                parsed = json.loads(json_str)
                
                if "ordered_entries" in parsed and isinstance(parsed["ordered_entries"], list):
                    json_candidates.append(parsed)
                    print(f"✓ JSON valido trovato e corretto con selector: {selector}")
                
            except json.JSONDecodeError as e:
                print(f"✗ JSONDecodeError con selector {selector}: {e}")
                print(f"  Primi 200 caratteri: {json_str[:200]}")
                continue
            except Exception as e:
                print(f"✗ Errore generico con selector {selector}: {e}")
                continue
    
    if json_candidates:
        print(f"Trovati {len(json_candidates)} JSON validi, ritorno l'ultimo")
        return json_candidates[-1]
    else:
        # Fallback: prova a estrarre da tutto il messaggio
        print("Tentativo fallback: estrazione da tutto il messaggio")
        full_text = page.locator("div[data-message-author-role='assistant']").first.evaluate("el => el.textContent")
        
        try:
            start = full_text.index("{")
            end = full_text.rindex("}") + 1
            json_str = full_text[start:end]
            json_str = json_str.replace('\u200b', '').replace('\xa0', ' ')
            
            # Applica la stessa correzione
            json_str = re.sub(r'\{\s*"(tt\d+)"\s*\}', r'"\1"', json_str)
            
            parsed = json.loads(json_str)
            print("✓ JSON trovato e corretto con fallback")
            return parsed
        except Exception as e:
            print(f"✗ Fallback fallito: {e}")
            raise ValueError("Nessun JSON valido trovato nella pagina")

def process_query_with_retries(page, full_prompt, idx, max_retries=MAX_RETRIES):
    """
    Process a query with retry logic. Returns parsed result or None if all retries fail.
    """
    for attempt in range(max_retries):
        try:
            print(f"🔄 Tentativo {attempt + 1}/{max_retries} per riga {idx}...")
            
            # Conta messaggi esistenti
            prev_count = page.locator("div[data-message-author-role='assistant']").count()

            # Invio prompt
            textarea = page.locator('div.ProseMirror[contenteditable="true"]')
            textarea.wait_for(state="visible", timeout=60000)
            textarea.fill(full_prompt)
            textarea.press("Enter")

            time.sleep(30)
            
            # Attendi nuova risposta
            try:
                page.wait_for_function(
                    f"document.querySelectorAll('div[data-message-author-role=\"assistant\"]').length > {prev_count}",
                    timeout=120000
                )
                page.wait_for_timeout(random.randint(*WAIT_AFTER_REPLY) * 1000)  # attesa tra 20 e 25 sec
            except:
                print(f"⏳ Timeout risposta su tentativo {attempt + 1}")
                if attempt == max_retries - 1:  # Last attempt
                    raise Exception("Timeout finale dopo tutti i tentativi")
                continue
            
            time.sleep(10)
            
            # Estrai e valida JSON
            
            parsed = extract_json_from_page(page)
            if "ordered_entries" in parsed and isinstance(parsed["ordered_entries"], list):
                # Unisce tutti gli elementi della lista con il separatore "|"
                result_to_save = "|".join(parsed["ordered_entries"])
                print(f"✅ Tentativo {attempt + 1} riuscito per riga {idx}")
                return result_to_save
            else:
                raise ValueError("'ordered_entries' missing or invalid")
                
        except Exception as e:
            print(f"❌ Errore tentativo {attempt + 1}/{max_retries} per riga {idx}: {e}")
            if attempt < max_retries - 1:
                print(f"🔄 Riprovo...")
                time.sleep(5)  # Brief pause before retry
            else:
                print(f"💀 Tutti i tentativi falliti per riga {idx}. Salvataggio risultato vuoto.")
                return []
    
    return []


def extract_movie_ids(response: str) -> list:
    """
    Estrae gli ID dei film dalla risposta di Gemini.

    Args:
        response: risposta di Gemini

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

    print(f"Calcolo metriche per predicted_ids: {predicted_ids}, ground_truth: {ground_truth_str}, nl_query: {nl_query}")
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



with sync_playwright() as p:
    browser, page = launch_browser(p)
    input("Premi Invio dopo aver fatto login...")
    send_system_message(page)

    count_global = 0

    

    for test in tests_list:
        test_path = test["test_path"]
        out_path = test["out_path"]
        exec_modes = test.get("modes", ["NORMAL", "NULL", "REMOVE"])

        for EXEC_TYPE in exec_modes:
            
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            out_path = f"{test['out_path']}/gpt_{EXEC_TYPE}_{test['out_name']}"

            last_completed_idx = int(progress.get(f"{test_path}_{EXEC_TYPE}", -1))
            df = pd.read_csv(test_path, dtype=str).dropna(subset=['nl_query'])
            df = df[df.index > last_completed_idx]

            if df.empty:
                print(f"[SKIP] {test_path} ({EXEC_TYPE}) già completato.")
                continue

            for idx, row in df.iterrows():
                try:
                    # Riavvio browser periodico
                    if count_global > 0 and count_global % RESTART_BROWSER_EVERY == 0:
                        print(">>> Riavvio browser per evitare rallentamenti...")
                        count_global = 0
                        browser.close()
                        browser, page = launch_browser(p)
                        send_system_message(page)

                    # Reset chat periodico
                    elif count_global > 0 and count_global % RESET_CHAT_EVERY == 0:
                        print(">>> Nuova chat per evitare DOM enorme...")
                        time.sleep(40)
                        page.wait_for_selector('a[data-testid="create-new-chat-button"]', timeout=60000)
                        page.click('a[data-testid="create-new-chat-button"]', force=True)
                        page.wait_for_timeout(2000)
                        send_system_message(page)

                    csv_path = row['csv_path']
                    nl_query = row['nl_query']
                    
                    # Load all tables needed for this query

                    tables_data = load_tables_for_query(csv_path, EXEC_TYPE)
                    

                    full_prompt = f"""
                                         Instructions:
                                        - Analyze the movies in the dataset
                                        - For similarity requests, find movies with similar genres, themes, or characteristics
                                        - For negative similarity requests ("opposto", "opposite", "contrario"), find movies that are opposite in style, genre, or theme
                                        - Return the results as a list of movie IDs (tt codes) that best match the request
                                        - Consider factors like genre, year, rating, plot, and other available attributes when available
                                        - Provide exactly 10 recommendations when possible
                                        - If some data is missing (NULL values), use the available information to make the best recommendations

                                        Dataset:
                                        {tables_data}

                                        User Request:
                                        {nl_query}

                                    
                                        Response Format:
                                        Return only the movie IDs (tt codes),in json format:
                                        "ordered_entries": ["tt0123456", "tt0123456", "tt0444444"]
                                        
                                    """
                                        

                    # Process query with retry logic
                    result_to_save = process_query_with_retries(page, full_prompt, idx)

                    all_metrics = calculate_enhanced_metrics(result_to_save.split('|'), row.get('ground_truth', ''), 
                                           row['nl_query'], df)
                    
                    print(f"Metriche calcolate: {all_metrics}")

                    # Check if browser needs to be restarted after failed retries
                    if not result_to_save and count_global > 0:
                        print("🔄 Browser restart after failed retries...")
                        browser.close()
                        browser, page = launch_browser(p)
                        send_system_message(page)
                   
                    row_result = {
                        "csv_path": csv_path,
                        "test_language": row.get("test_language", ""),
                        "test_category": row.get("test_category", ""),
                        "ground_truth": row.get("ground_truth", ""),
                        "nl_query": nl_query,
                        "test_id": row.get("test_id", ""),
                        "model": model,
                        "execu_type": EXEC_TYPE,                  
                        "result": result_to_save,  
                        "retry_count": MAX_RETRIES if not result_to_save else 1,
                        **all_metrics
                    }

                    

                    pd.DataFrame([row_result]).to_csv(out_path, mode='a', index=False, header=not os.path.exists(out_path))
                    progress[f"{test_path}_{EXEC_TYPE}"] = idx
                    with open(progress_path, 'w') as f:
                        json.dump(progress, f)

                    count_global += 1
                    
                    gc.collect()

                except Exception as e:
                    print(f"❌ Errore critico sulla riga {idx}: {e}")
                    # Save empty result for critical errors too
                    row_result = {
                        "csv_path": csv_path,
                        "test_language": row.get("test_language", ""),
                        "test_category": row.get("test_category", ""),
                        "ground_truth": row.get("ground_truth", ""),
                        "nl_query": nl_query,
                        "test_id": row.get("test_id", ""),
                        "model": model,
                        "execu_type": EXEC_TYPE,                  
                        "result": '',  
                        "retry_count": MAX_RETRIES if not result_to_save else 1,
                        # Metriche base
                        'precision': 0, 'recall': 0, 'f1': 0, 'accuracy': 0,
                        'intersection_size': 0, 'ground_truth_size': 0, 'predicted_size': 0,
                        # Metriche avanzate
                        'is_negative_query': False, 'ndcg_10': 0.0, 'precision_5': 0.0, 'precision_10': 0.0,
                        'spearman_correlation': 0.0, 'diversity_score': 0.0,
                        'genre_opposition_rate_5': 0.0, 'genre_opposition_rate_10': 0.0
                    }
                    pd.DataFrame([row_result]).to_csv(out_path, mode='a', index=False, header=not os.path.exists(out_path))
                    progress[f"{test_path}_{EXEC_TYPE}"] = idx
                    with open(progress_path, 'w') as f:
                        json.dump(progress, f)
                    continue

    if browser:
        browser.close()
