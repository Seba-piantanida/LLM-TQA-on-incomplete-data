"""
Generatore di test cases per similarità film bilingue (IT/EN) con casi positivi e negativi

Questo script:
1. Carica un dataset IMDB nel formato specificato
2. Calcola similarità deterministiche tra film
3. Genera test cases con query in italiano e inglese
4. Include sia similarità positive che negative
5. Esporta nel formato CSV richiesto: csv_path, test_language, test_category, ground_truth, nl_query

Uso: python generate_bilingual_similarity_tests.py
"""

import pandas as pd
import numpy as np
import random
from typing import List, Dict, Tuple
import os

def parse_list_field(field_value):
    """Converte stringa separata da virgola in lista"""
    if pd.isna(field_value) or field_value == '':
        return []
    return [item.strip() for item in str(field_value).split(',')]

def calculate_decade(year):
    """Calcola il decennio"""
    return (year // 10) * 10

def calculate_similarity_score(movie1, movie2, weights=None):
    """
    Calcola punteggio di similarità deterministico

    Componenti:
    - Regista: +3 punti
    - Genere primario: +2 punti  
    - Decade: +1 punto
    - Generi comuni: +1 per genere
    - Cast condiviso: +2 per attore
    - Scrittore comune: +1 per scrittore
    - Durata simile: +1 se ±30 min
    - Rating simile: +1 se ±0.5
    """
    if weights is None:
        weights = {
            'regista': 3,
            'genere_primario': 2,
            'decade': 1,
            'generi_comuni': 1,
            'cast_condiviso': 2,
            'scrittore_comune': 1,
            'durata_simile': 1,
            'rating_simile': 1
        }

    score = 0
    details = {}

    # 1. Regista principale
    registi1 = parse_list_field(movie1['directors'])
    registi2 = parse_list_field(movie2['directors'])
    regista1_main = registi1[0] if registi1 else None
    regista2_main = registi2[0] if registi2 else None

    if regista1_main and regista2_main and regista1_main == regista2_main:
        score += weights['regista']
        details['regista'] = weights['regista']
    else:
        details['regista'] = 0

    # 2. Genere primario
    generi1 = parse_list_field(movie1['genre'])
    generi2 = parse_list_field(movie2['genre'])
    genere1_main = generi1[0] if generi1 else None
    genere2_main = generi2[0] if generi2 else None

    if genere1_main and genere2_main and genere1_main == genere2_main:
        score += weights['genere_primario']
        details['genere_primario'] = weights['genere_primario']
    else:
        details['genere_primario'] = 0

    # 3. Decade
    decade1 = calculate_decade(movie1['year'])
    decade2 = calculate_decade(movie2['year'])
    if decade1 == decade2:
        score += weights['decade']
        details['decade'] = weights['decade']
    else:
        details['decade'] = 0

    # 4. Generi in comune
    generi_overlap = set(generi1) & set(generi2)
    if len(generi_overlap) > 1:
        bonus = weights['generi_comuni'] * (len(generi_overlap) - 1)
        score += bonus
        details['generi_comuni'] = bonus
    else:
        details['generi_comuni'] = 0

    # 5. Cast condiviso
    cast1 = parse_list_field(movie1['main_cast'])
    cast2 = parse_list_field(movie2['main_cast'])
    cast_overlap = set(cast1) & set(cast2)
    if cast_overlap:
        bonus = weights['cast_condiviso'] * len(cast_overlap)
        score += bonus
        details['cast_condiviso'] = bonus
    else:
        details['cast_condiviso'] = 0

    # 6. Scrittori
    scrittori1 = parse_list_field(movie1['writers'])
    scrittori2 = parse_list_field(movie2['writers'])
    scrittori_overlap = set(scrittori1) & set(scrittori2)
    if scrittori_overlap:
        bonus = weights['scrittore_comune'] * len(scrittori_overlap)
        score += bonus
        details['scrittore_comune'] = bonus
    else:
        details['scrittore_comune'] = 0

    # 7. Durata simile
    durata_diff = abs(movie1['duration_min'] - movie2['duration_min'])
    if durata_diff <= 30:
        score += weights['durata_simile']
        details['durata_simile'] = weights['durata_simile']
    else:
        details['durata_simile'] = 0

    # 8. Rating simile
    rating_diff = abs(movie1['AVG_score'] - movie2['AVG_score'])
    if rating_diff <= 0.5:
        score += weights['rating_simile']
        details['rating_simile'] = weights['rating_simile']
    else:
        details['rating_simile'] = 0

    return score, details

def generate_positive_queries_italian(movie):
    """Genera query positive in italiano per un film"""
    titolo = movie['title']
    anno = movie['year']
    regista = parse_list_field(movie['directors'])[0] if parse_list_field(movie['directors']) else None
    generi = parse_list_field(movie['genre'])
    genere_principale = generi[0] if generi else None

    queries = [
        f"Trova film simili a '{titolo}'",
        f"Consiglia film come '{titolo}' del {anno}",
        f"Suggerisci film simili a '{titolo}'",
        f"Altri film come '{titolo}' ({anno})",
        f"Raccomandazioni basate su '{titolo}'",
        f"Se mi è piaciuto '{titolo}', cosa dovrei guardare?",
        f"Film con stile simile a '{titolo}' del {anno}",
        f"Pellicole che ricordano '{titolo}'",
        f"Ho amato '{titolo}', raccomandami qualcosa di simile",
        f"Cerco film del genere di '{titolo}'"
    ]

    if regista:
        queries.extend([
            f"Film simili a '{titolo}' di {regista}",
            f"Altri film di {regista} simili a '{titolo}'"
        ])

    if genere_principale:
        queries.append(f"Film {genere_principale.lower()} simili a '{titolo}'")

    return queries

def generate_negative_queries_italian(movie):
    """Genera query negative in italiano per un film"""
    titolo = movie['title']
    anno = movie['year']
    
    queries = [
        f"Ho odiato '{titolo}', raccomandami film che potrebbero piacermi",
        f"'{titolo}' non mi è piaciuto per niente, cosa altro posso guardare?",
        f"Detesto film come '{titolo}', suggerisci qualcosa di completamente diverso",
        f"'{titolo}' è stato terribile, consiglia l'opposto",
        f"Non sopporto '{titolo}', trova film che non gli assomigliano",
        f"Mi ha fatto schifo '{titolo}', raccomanda qualcosa di diverso",
        f"'{titolo}' del {anno} mi ha deluso, cosa guardare invece?",
        f"Ho trovato '{titolo}' noioso, suggerisci qualcosa di più interessante",
        f"'{titolo}' non fa per me, raccomandazioni di genere opposto?",
        f"Evito film come '{titolo}', cosa mi consiglieresti?"
    ]
    
    return queries

def generate_positive_queries_english(movie):
    """Genera query positive in inglese per un film"""
    title = movie['title']
    year = movie['year']
    director = parse_list_field(movie['directors'])[0] if parse_list_field(movie['directors']) else None
    genres = parse_list_field(movie['genre'])
    main_genre = genres[0] if genres else None

    queries = [
        f"Find movies similar to '{title}'",
        f"Recommend movies like '{title}' from {year}",
        f"Suggest films similar to '{title}'",
        f"Other movies like '{title}' ({year})",
        f"Recommendations based on '{title}'",
        f"If I liked '{title}', what should I watch?",
        f"Movies with similar style to '{title}' from {year}",
        f"Films that remind me of '{title}'",
        f"I loved '{title}', recommend something similar",
        f"Looking for movies in the vein of '{title}'"
    ]

    if director:
        queries.extend([
            f"Movies similar to '{title}' by {director}",
            f"Other {director} films like '{title}'"
        ])

    if main_genre:
        queries.append(f"{main_genre} movies similar to '{title}'")

    return queries

def generate_negative_queries_english(movie):
    """Genera query negative in inglese per un film"""
    title = movie['title']
    year = movie['year']
    
    queries = [
        f"I hated '{title}', recommend movies I might like",
        f"'{title}' wasn't for me, what else can I watch?",
        f"I despise movies like '{title}', suggest something completely different",
        f"'{title}' was terrible, recommend the opposite",
        f"Can't stand '{title}', find movies that don't resemble it",
        f"'{title}' disgusted me, recommend something different",
        f"'{title}' from {year} disappointed me, what to watch instead?",
        f"Found '{title}' boring, suggest something more interesting",
        f"'{title}' isn't my type, recommendations for opposite genre?",
        f"I avoid movies like '{title}', what would you recommend?"
    ]
    
    return queries

def get_similar_movies(anchor_movie, dataset, top_n=10):
    """Ottiene i film più simili"""
    anchor_idx = anchor_movie.name
    
    similarities = []
    for i, movie in dataset.iterrows():
        if i != anchor_idx:
            score, details = calculate_similarity_score(anchor_movie, movie)
            if score > 0:  # Solo film con similarità positiva
                similarities.append({
                    'IMDB_id': movie['IMDB_id'],
                    'title': movie['title'],
                    'score': score
                })
    
    # Ordina per score decrescente
    similarities.sort(key=lambda x: x['score'], reverse=True)
    return similarities[:top_n]

def get_dissimilar_movies(anchor_movie, dataset, top_n=10):
    """Ottiene i film meno simili (per query negative)"""
    anchor_idx = anchor_movie.name
    anchor_genres = set(parse_list_field(anchor_movie['genre']))
    anchor_decade = calculate_decade(anchor_movie['year'])
    anchor_directors = set(parse_list_field(anchor_movie['directors']))
    
    dissimilar_candidates = []
    
    for i, movie in dataset.iterrows():
        if i != anchor_idx:
            # Calcola "dissimilarità"
            movie_genres = set(parse_list_field(movie['genre']))
            movie_decade = calculate_decade(movie['year'])
            movie_directors = set(parse_list_field(movie['directors']))
            
            dissimilarity_score = 0
            
            # Generi diversi
            if not (anchor_genres & movie_genres):
                dissimilarity_score += 3
                
            # Decenni distanti
            decade_diff = abs(anchor_decade - movie_decade)
            if decade_diff >= 20:
                dissimilarity_score += 2
                
            # Registi diversi
            if not (anchor_directors & movie_directors):
                dissimilarity_score += 2
                
            # Rating molto diverso
            rating_diff = abs(anchor_movie['AVG_score'] - movie['AVG_score'])
            if rating_diff >= 2.0:
                dissimilarity_score += 1
                
            # Durata molto diversa
            duration_diff = abs(anchor_movie['duration_min'] - movie['duration_min'])
            if duration_diff >= 60:
                dissimilarity_score += 1
                
            if dissimilarity_score > 0:
                dissimilar_candidates.append({
                    'IMDB_id': movie['IMDB_id'],
                    'title': movie['title'],
                    'dissimilarity_score': dissimilarity_score
                })
    
    # Ordina per dissimilarità decrescente
    dissimilar_candidates.sort(key=lambda x: x['dissimilarity_score'], reverse=True)
    return dissimilar_candidates[:top_n]

import argparse
import os

def main():
    """Funzione principale"""
    parser = argparse.ArgumentParser(
        description="🎬 Generatore Test Cases Similarità Film Bilingue"
    )

    # Parametro obbligatorio: CSV di input
    parser.add_argument(
        "csv_filename",
        type=str,
        help="Percorso del file CSV di input contenente i dati IMDB"
    )

    # Parametri opzionali con default
    parser.add_argument(
        "--max_tests",
        type=int,
        default=200,
        help="Numero massimo totale di test da generare (default: 200)"
    )

    parser.add_argument(
        "--tests_per_category",
        type=int,
        default=None,
        help="Numero di test per categoria (default: max_tests // 4)"
    )

    args = parser.parse_args()

    csv_filename = args.csv_filename
    max_tests = args.max_tests
    tests_per_category = args.tests_per_category or max_tests // 4

    print("🎬 Generatore Test Cases Similarità Film Bilingue")
    print("=" * 60)
    print(f"CSV di input: {csv_filename}")
    print(f"Numero massimo di test: {max_tests}")
    print(f"Test per categoria: {tests_per_category}")

    # Verifica esistenza file
    if not os.path.exists(csv_filename):
        print(f"❌ File {csv_filename} non trovato")
        return
    # Carica dataset
    try:
        df = pd.read_csv(csv_filename)
        print(f"✅ Dataset caricato: {len(df)} film")
        
        # Verifica colonne richieste
        required_columns = ['IMDB_id', 'title', 'year', 'genre', 'directors', 'writers', 'main_cast', 'duration_min', 'AVG_score', 'number_of_votes']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"❌ Colonne mancanti nel CSV: {missing_columns}")
            return
            
    except Exception as e:
        print(f"❌ Errore nel caricamento del CSV: {e}")
        return

    # Pulisci dati
    df = df.dropna(subset=['title', 'year', 'genre', 'directors'])
    print(f"✅ Dataset pulito: {len(df)} film validi")

    if len(df) < 20:
        print("❌ Dataset troppo piccolo per generare test significativi")
        return

    # Seleziona film anchor casualmente
    anchor_movies = df.sample(n=min(tests_per_category, len(df)), random_state=42)
    print(f"📍 Selezionati {len(anchor_movies)} film anchor")

    # Lista per i risultati
    test_results = []

    print("\n🔄 Generazione test cases...")
    
    for i, (_, anchor_movie) in enumerate(anchor_movies.iterrows()):
        if i % 10 == 0:
            print(f"   Progresso: {i}/{len(anchor_movies)}")

        # POSITIVE SIMILARITY - ITALIAN
        similar_movies = get_similar_movies(anchor_movie, df, top_n=10)
        if similar_movies:
            ground_truth = [movie['IMDB_id'] for movie in similar_movies]
            queries_it_pos = generate_positive_queries_italian(anchor_movie)
            
            # Seleziona 2 query casuali per questo film
            selected_queries = random.sample(queries_it_pos, min(2, len(queries_it_pos)))
            
            for query in selected_queries:
                test_results.append({
                    'csv_path': csv_filename,
                    'test_language': 'italian',
                    'test_category': 'positive_similarity',
                    'ground_truth': '|'.join(ground_truth),
                    'nl_query': query
                })

        # NEGATIVE SIMILARITY - ITALIAN  
        dissimilar_movies = get_dissimilar_movies(anchor_movie, df, top_n=10)
        if dissimilar_movies:
            ground_truth = [movie['IMDB_id'] for movie in dissimilar_movies]
            queries_it_neg = generate_negative_queries_italian(anchor_movie)
            
            selected_queries = random.sample(queries_it_neg, min(2, len(queries_it_neg)))
            
            for query in selected_queries:
                test_results.append({
                    'csv_path': csv_filename,
                    'test_language': 'italian',
                    'test_category': 'negative_similarity',
                    'ground_truth': '|'.join(ground_truth),
                    'nl_query': query
                })

        # POSITIVE SIMILARITY - ENGLISH
        if similar_movies:
            ground_truth = [movie['IMDB_id'] for movie in similar_movies]
            queries_en_pos = generate_positive_queries_english(anchor_movie)
            
            selected_queries = random.sample(queries_en_pos, min(2, len(queries_en_pos)))
            
            for query in selected_queries:
                test_results.append({
                    'csv_path': csv_filename,
                    'test_language': 'english',
                    'test_category': 'positive_similarity',
                    'ground_truth': '|'.join(ground_truth),
                    'nl_query': query
                })

        # NEGATIVE SIMILARITY - ENGLISH
        if dissimilar_movies:
            ground_truth = [movie['IMDB_id'] for movie in dissimilar_movies]
            queries_en_neg = generate_negative_queries_english(anchor_movie)
            
            selected_queries = random.sample(queries_en_neg, min(2, len(queries_en_neg)))
            
            for query in selected_queries:
                test_results.append({
                    'csv_path': csv_filename,
                    'test_language': 'english',
                    'test_category': 'negative_similarity',
                    'ground_truth': '|'.join(ground_truth),
                    'nl_query': query
                })

    # Limita a max_tests se necessario
    if len(test_results) > max_tests:
        test_results = random.sample(test_results, max_tests)

    # Crea DataFrame e salva
    df_output = pd.DataFrame(test_results)
    output_filename = 'movie_similarity_bilingual_tests.csv'
    df_output.to_csv(output_filename, index=False, encoding='utf-8')

   

if __name__ == "__main__":
    main()