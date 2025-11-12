#!/usr/bin/env python3
"""
Script ottimizzato per creare un dataset IMDB consolidato dai file TSV ufficiali
con filtro integrato per film validi (title, year, genre, directors obbligatori)

Questo script ottimizzato per memoria:
1. Carica i file in chunks per ridurre l'uso di RAM
2. Filtra i dati progressivamente per mantenere solo ciò che serve
3. Usa tipi di dati ottimizzati per ridurre l'occupazione memoria
4. Processa i dati in batch per evitare overflow di memoria
5. Garantisce che tutti i film finali abbiano dati essenziali completi

Autore: Pipeline Test Framework
Data: Settembre 2025
"""

import pandas as pd
import numpy as np
import gzip
from typing import Dict, List, Optional
import warnings
import gc
import os
warnings.filterwarnings('ignore')

# Configurazione per ottimizzare memoria
CHUNK_SIZE = 10000  # Dimensione chunk per lettura file
MAX_MOVIES = 4000   # Numero massimo di film finali
MEMORY_SAVE_MODE = True  # Modalità risparmio memoria

def optimize_dataframe_memory(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ottimizza i tipi di dati del DataFrame per ridurre l'uso di memoria
    """
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            # Numeri interi
            if str(col_type).startswith('int'):
                if df[col].min() >= 0:  # Solo positivi
                    if df[col].max() < 255:
                        df[col] = df[col].astype(np.uint8)
                    elif df[col].max() < 65535:
                        df[col] = df[col].astype(np.uint16)
                    elif df[col].max() < 4294967295:
                        df[col] = df[col].astype(np.uint32)
                else:  # Con negativi
                    if df[col].min() > -128 and df[col].max() < 127:
                        df[col] = df[col].astype(np.int8)
                    elif df[col].min() > -32768 and df[col].max() < 32767:
                        df[col] = df[col].astype(np.int16)
            
            # Numeri decimali
            elif str(col_type).startswith('float'):
                if df[col].min() >= 0 and df[col].max() < 3.4e38:
                    df[col] = df[col].astype(np.float32)
    
    return df

def load_imdb_dataset_chunked(filename: str, 
                             required_columns: List[str] = None,
                             filter_func = None,
                             max_rows: int = None) -> pd.DataFrame:
    """
    Carica un dataset IMDB in modo ottimizzato per memoria usando chunks
    
    Args:
        filename: Nome del file TSV
        required_columns: Colonne da mantenere (None = tutte)
        filter_func: Funzione per filtrare i dati durante il caricamento
        max_rows: Numero massimo di righe da caricare
    
    Returns:
        DataFrame ottimizzato
    """
    print(f"Caricamento ottimizzato {filename}...")
    
    if not os.path.exists(filename):
        # Prova versione non compressa
        filename_uncompressed = filename.replace('.gz', '')
        if not os.path.exists(filename_uncompressed):
            print(f"❌ File {filename} non trovato")
            return pd.DataFrame()
        filename = filename_uncompressed
    
    chunks = []
    total_rows = 0
    
    try:
        # Leggi in chunks per ottimizzare memoria
        chunk_reader = pd.read_csv(
            filename, 
            sep='\t', 
            na_values=['\\N'], 
            chunksize=CHUNK_SIZE,
            low_memory=False
        )
        
        for i, chunk in enumerate(chunk_reader):
            if max_rows and total_rows >= max_rows:
                break
                
            # Applica filtro se fornito
            if filter_func:
                chunk = filter_func(chunk)
            
            # Mantieni solo colonne necessarie
            if required_columns:
                available_cols = [col for col in required_columns if col in chunk.columns]
                chunk = chunk[available_cols]
            
            # Ottimizza memoria del chunk
            if MEMORY_SAVE_MODE:
                chunk = optimize_dataframe_memory(chunk)
            
            if not chunk.empty:
                chunks.append(chunk)
                total_rows += len(chunk)
                
            if i % 10 == 0:
                print(f"   Processati {total_rows:,} record...")
                
        if not chunks:
            print(f"❌ Nessun dato caricato da {filename}")
            return pd.DataFrame()
            
        # Combina tutti i chunks
        result = pd.concat(chunks, ignore_index=True)
        
        # Pulizia memoria
        del chunks
        gc.collect()
        
        print(f"✅ Caricato {filename}: {len(result):,} righe, {len(result.columns)} colonne")
        return result
        
    except Exception as e:
        print(f"❌ Errore nel caricamento {filename}: {e}")
        return pd.DataFrame()

def create_name_lookup(name_basics_df: pd.DataFrame) -> Dict[str, str]:
    """
    Crea un dizionario per lookup veloce dei nomi
    """
    print("Creazione lookup nomi...")
    if name_basics_df.empty:
        return {}
    
    # Crea dizionario per lookup veloce
    name_dict = dict(zip(name_basics_df['nconst'], name_basics_df['primaryName']))
    print(f"   Creato lookup per {len(name_dict):,} persone")
    return name_dict

def extract_person_names_fast(person_ids: str, name_dict: Dict[str, str]) -> List[str]:
    """
    Versione ottimizzata per estrarre nomi usando dizionario di lookup
    """
    if pd.isna(person_ids) or person_ids == '\\N':
        return []
    
    names = []
    for person_id in person_ids.split(','):
        person_id = person_id.strip()
        if person_id in name_dict:
            names.append(name_dict[person_id])
    
    return names

def filter_recent_movies(chunk: pd.DataFrame) -> pd.DataFrame:
    """
    Filtra per tenere solo film recenti durante il caricamento
    """
    if 'titleType' not in chunk.columns:
        return chunk
        
    # Filtra solo film non adult con dati di base validi
    filtered = chunk[
        (chunk['titleType'] == 'movie') & 
        (chunk['isAdult'] == 0) &
        (chunk['startYear'].notna()) &
        (chunk['startYear'] != '\\N') &
        (chunk['primaryTitle'].notna()) &
        (chunk['primaryTitle'] != '\\N') &
        (chunk['primaryTitle'] != '') &
        (chunk['genres'].notna()) &
        (chunk['genres'] != '\\N') &
        (chunk['genres'] != '')
    ].copy()
    
    if not filtered.empty:
        # Converti anno e filtra per recenti (ultimi 30 anni circa)
        filtered['startYear'] = pd.to_numeric(filtered['startYear'], errors='coerce')
        filtered = filtered[
            (filtered['startYear'] >= 1990) & 
            (filtered['startYear'] <= 2025)
        ]
    
    return filtered

def is_valid_film(row: pd.Series, name_dict: Dict[str, str]) -> bool:
    """
    Verifica se un film ha tutti i dati essenziali
    """
    # Controlla titolo
    if pd.isna(row.get('primaryTitle')) or row.get('primaryTitle') == '' or row.get('primaryTitle') == '\\N':
        return False
    
    # Controlla anno
    if pd.isna(row.get('startYear')) or row.get('startYear') == '\\N':
        return False
    
    # Controlla generi
    if pd.isna(row.get('genres')) or row.get('genres') == '' or row.get('genres') == '\\N':
        return False
    
    # Controlla registi
    directors = row.get('directors')
    if pd.isna(directors) or directors == '' or directors == '\\N':
        return False
    
    # Verifica che almeno un regista sia nel dizionario nomi
    director_names = extract_person_names_fast(directors, name_dict)
    if not director_names:
        return False
    
    return True

def create_consolidated_dataset_optimized(max_movies: int = 4000) -> pd.DataFrame:
    """
    Versione ottimizzata per memoria del creatore di dataset
    con filtro integrato per film validi
    """
    print("=== Creazione Dataset IMDB Consolidato (Ottimizzato con Filtro Validi) ===\n")
    
    # 1. Carica basics con filtro per film recenti
    print("1. Caricamento film (con pre-filtro per memoria)...")
    movies = load_imdb_dataset_chunked(
        'title.basics.tsv.gz',
        required_columns=['tconst', 'titleType', 'primaryTitle', 'isAdult', 
                         'startYear', 'runtimeMinutes', 'genres'],
        filter_func=filter_recent_movies,
        max_rows=100000  # Aumentato per compensare i filtri successivi
    )
    
    if movies.empty:
        print("ERRORE: Impossibile caricare film")
        return pd.DataFrame()
    
    print(f"Film pre-filtrati caricati: {len(movies):,}")
    
    # 2. Carica nomi PRIMA di filtrare i film per directors
    print("2. Caricamento nomi per validazione...")
    names = load_imdb_dataset_chunked(
        'name.basics.tsv.gz',
        required_columns=['nconst', 'primaryName'],
        max_rows=1000000  # Aumentato per avere più copertura
    )
    
    name_dict = create_name_lookup(names)
    del names
    gc.collect()
    
    # 3. Carica crew per tutti i film pre-selezionati
    print("3. Caricamento crew...")
    movie_ids = set(movies['tconst'])
    
    def filter_crew(chunk):
        return chunk[chunk['tconst'].isin(movie_ids)]
    
    crew = load_imdb_dataset_chunked(
        'title.crew.tsv.gz',
        required_columns=['tconst', 'directors', 'writers'],
        filter_func=filter_crew
    )
    
    # 4. Merge movies con crew
    if not crew.empty:
        movies_with_crew = movies.merge(crew, on='tconst', how='inner')  # INNER join per avere solo film con crew
    else:
        print("ERRORE: Nessun dato crew caricato")
        return pd.DataFrame()
    
    print(f"Film con crew: {len(movies_with_crew):,}")
    
    # 5. Filtra per film VALIDI usando la funzione di validazione
    print("4. Filtro per film validi (titolo + anno + generi + registi)...")
    valid_movies = []
    
    for idx, row in movies_with_crew.iterrows():
        if is_valid_film(row, name_dict):
            valid_movies.append(row)
            
        if len(valid_movies) % 1000 == 0 and len(valid_movies) > 0:
            print(f"   Film validi trovati: {len(valid_movies):,}")
    
    if not valid_movies:
        print("ERRORE: Nessun film valido trovato")
        return pd.DataFrame()
    
    movies_valid = pd.DataFrame(valid_movies)
    print(f"✅ Film con tutti i dati essenziali: {len(movies_valid):,}")
    
    del movies, movies_with_crew, crew
    gc.collect()
    
    # 6. Carica ratings solo per film validi
    print("5. Caricamento ratings per film validi...")
    valid_movie_ids = set(movies_valid['tconst'])
    
    def filter_ratings(chunk):
        return chunk[chunk['tconst'].isin(valid_movie_ids)]
    
    ratings = load_imdb_dataset_chunked(
        'title.ratings.tsv.gz',
        required_columns=['tconst', 'averageRating', 'numVotes'],
        filter_func=filter_ratings
    )
    
    # 7. Merge con ratings e seleziona top film
    if not ratings.empty:
        movies_with_ratings = movies_valid.merge(ratings, on='tconst', how='left')
        # Ordina per rating*voti per avere film di qualità
        movies_with_ratings['score'] = (
            movies_with_ratings['averageRating'].fillna(6.0) * 
            np.log1p(movies_with_ratings['numVotes'].fillna(100))
        )
        movies_final = movies_with_ratings.nlargest(max_movies, 'score')
    else:
        # Se non ci sono ratings, ordina per anno (più recenti prima)
        movies_final = movies_valid.nlargest(max_movies, 'startYear')
    
    print(f"Film finali selezionati: {len(movies_final):,}")
    
    del ratings, movies_valid
    gc.collect()
    
    # 8. Cast principale (versione semplificata per memoria)
    print("6. Cast principale (versione ottimizzata)...")
    final_movie_ids = set(movies_final['tconst'])
    
    # Carica solo principals necessari
    def filter_principals(chunk):
        return chunk[
            chunk['tconst'].isin(final_movie_ids) & 
            chunk['category'].isin(['actor', 'actress']) &
            (chunk['ordering'] <= 5)  # Solo primi 5 per memoria
        ]
    
    principals = load_imdb_dataset_chunked(
        'title.principals.tsv.gz',
        required_columns=['tconst', 'nconst', 'category', 'ordering'],
        filter_func=filter_principals
    )
    
    # Elabora cast
    cast_dict = {}
    if not principals.empty:
        for tconst in final_movie_ids:
            film_cast = principals[
                (principals['tconst'] == tconst)
            ].sort_values('ordering').head(5)
            
            cast_names = []
            for nconst in film_cast['nconst']:
                if nconst in name_dict:
                    cast_names.append(name_dict[nconst])
            
            cast_dict[tconst] = cast_names
    
    # 9. Elabora nomi registi e scrittori
    print("7. Elaborazione nomi finali...")
    
    # Elabora registi (garantiti validi dal filtro precedente)
    movies_final['director_names'] = movies_final['directors'].apply(
        lambda x: extract_person_names_fast(x, name_dict)
    )
    
    # Elabora scrittori  
    movies_final['writer_names'] = movies_final['writers'].apply(
        lambda x: extract_person_names_fast(x, name_dict)
    )
    
    # Aggiungi cast
    movies_final['main_cast'] = movies_final['tconst'].map(
        lambda x: cast_dict.get(x, [])
    )
    
    # 10. Dataset finale ottimizzato
    print("8. Finalizzazione...")
    final_dataset = pd.DataFrame({
        'film_id': movies_final['tconst'],
        'title': movies_final['primaryTitle'],  # Cambiato da 'titolo' a 'title'
        'year': movies_final['startYear'].astype('int16'),  # Cambiato da 'anno' a 'year'
        'genre': movies_final['genres'].fillna('').apply(  # Cambiato da 'generi' a 'genre'
            lambda x: x.replace(',', '|') if x else ''
        ),
        'directors': movies_final['director_names'].apply(  # Cambiato da 'registi' a 'directors'
            lambda x: '|'.join(x) if isinstance(x, list) else ''
        ),
        'writers': movies_final['writer_names'].apply(  # Cambiato da 'scrittori' a 'writers'
            lambda x: '|'.join(x) if isinstance(x, list) else ''
        ),
        'main_cast': movies_final['main_cast'].apply(
            lambda x: '|'.join(x) if isinstance(x, list) else ''
        ),
        'runtime_minutes': movies_final['runtimeMinutes'].fillna(0).astype('int16'),
        'average_rating': movies_final['averageRating'].fillna(0.0).astype('float32'),
        'num_votes': movies_final['numVotes'].fillna(0).astype('int32')
    })
    
    # 11. VALIDAZIONE FINALE: controlla che tutti i film abbiano i dati essenziali
    print("9. Validazione finale dati essenziali...")
    initial_count = len(final_dataset)
    
    # Applica il filtro finale per essere sicuri
    final_dataset = final_dataset.dropna(subset=['title', 'year', 'genre', 'directors'])
    final_dataset = final_dataset[
        (final_dataset['title'] != '') & 
        (final_dataset['genre'] != '') &
        (final_dataset['directors'] != '') &
        (final_dataset['year'] > 1900)
    ]
    
    final_count = len(final_dataset)
    print(f"✅ Dataset pulito: {final_count} film validi (rimossi {initial_count - final_count} film incompleti)")
    
    # Ordina per anno decrescente
    final_dataset = final_dataset.sort_values('year', ascending=False)
    
    # Pulizia memoria finale
    gc.collect()
    
    return final_dataset

def display_memory_usage():
    """Mostra l'uso attuale della memoria"""
    try:
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        print(f"💾 Memoria attualmente utilizzata: {memory_mb:.1f} MB")
    except ImportError:
        print("💾 Modulo psutil non disponibile per monitoraggio memoria")

def display_dataset_info(df: pd.DataFrame):
    """Mostra informazioni di riepilogo sul dataset"""
    print(f"\n=== Informazioni Dataset ===")
    print(f"Numero totale film: {len(df):,}")
    print(f"Colonne: {', '.join(df.columns)}")
    print(f"\nAnno minimo: {df['year'].min()}")
    print(f"Anno massimo: {df['year'].max()}")
    print(f"Durata media: {df['runtime_minutes'].mean():.1f} minuti")
    print(f"Rating medio: {df['average_rating'].mean():.2f}")
    
    print(f"\nValidità dati (tutti i film dovrebbero avere questi):")
    print(f"- Con titolo: {len(df[df['title'] != ''])}")
    print(f"- Con anno: {len(df[df['year'] > 0])}")
    print(f"- Con generi: {len(df[df['genre'] != ''])}")
    print(f"- Con registi: {len(df[df['directors'] != ''])}")
    
    print(f"\nInformazioni aggiuntive:")
    print(f"- Con scrittori: {len(df[df['writers'] != ''])}")  
    print(f"- Con cast: {len(df[df['main_cast'] != ''])}")
    print(f"- Con rating: {len(df[df['average_rating'] > 0])}")
    
    # Verifica che non ci siano valori mancanti nei campi essenziali
    missing_title = df[df['title'].isna() | (df['title'] == '')]['title'].count()
    missing_year = df[df['year'].isna() | (df['year'] <= 1900)]['year'].count()
    missing_genre = df[df['genre'].isna() | (df['genre'] == '')]['genre'].count()
    missing_directors = df[df['directors'].isna() | (df['directors'] == '')]['directors'].count()
    
    if missing_title == 0 and missing_year == 0 and missing_genre == 0 and missing_directors == 0:
        print(f"✅ VALIDAZIONE SUPERATA: Tutti i film hanno title, year, genre e directors")
    else:
        print(f"⚠ ATTENZIONE: Trovati film con dati mancanti:")
        if missing_title > 0: print(f"   - Titoli mancanti: {missing_title}")
        if missing_year > 0: print(f"   - Anni mancanti/invalidi: {missing_year}")
        if missing_genre > 0: print(f"   - Generi mancanti: {missing_genre}")
        if missing_directors > 0: print(f"   - Registi mancanti: {missing_directors}")

def main():
    """Funzione principale ottimizzata"""
    print("Script IMDB Consolidato - Versione Ottimizzata con Filtro Film Validi")
    print("=" * 70)
    print("\n🚀 Ottimizzazioni attive:")
    print(f"- Lettura in chunks da {CHUNK_SIZE:,} righe")
    print(f"- Pre-filtro per film recenti (dal 1990)")
    print(f"- Filtro integrato per film VALIDI (title + year + genre + directors)")
    print(f"- Tipi di dati ottimizzati")
    print(f"- Gestione memoria con garbage collection")
    print(f"- Limite finale: {MAX_MOVIES:,} film validi")
    print("-" * 70)
    
    display_memory_usage()
    
    # Crea il dataset ottimizzato
    dataset = create_consolidated_dataset_optimized(max_movies=MAX_MOVIES)
    
    if dataset.empty:
        print("\n⚠ Errore: Impossibile creare il dataset.")
        return
    
    display_memory_usage()
    display_dataset_info(dataset)
    
    # Salva il dataset
    output_filename = 'imdb_dataset_consolidato_optimized_valid.csv'
    print(f"\nSalvataggio dataset in {output_filename}...")
    dataset.to_csv(output_filename, index=False, encoding='utf-8')
    print(f"✅ Dataset salvato con successo!")
    
    # Mostra alcune righe di esempio
    print(f"\n=== Prime 5 righe del dataset ===")
    print(dataset.head().to_string())
    
    display_memory_usage()
    print(f"\n🎉 Script completato con successo!")
    print(f"Dataset finale: {output_filename} ({len(dataset):,} film VALIDI)")

if __name__ == "__main__":
    main()