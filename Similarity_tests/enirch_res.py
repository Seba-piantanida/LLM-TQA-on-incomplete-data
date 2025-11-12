import pandas as pd
import numpy as np
import tiktoken
import logging
import os
from pathlib import Path
from tqdm import tqdm
import warnings
import ast

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

def parse_removed_columns(rem_col_str: str) -> list:
    """
    Estrae la lista di colonne rimosse dalla stringa rem_col.
    Gestisce sia formato lista Python ['col1', 'col2'] che stringa separata da virgole.
    
    Args:
        rem_col_str: stringa con colonne (formato lista Python o CSV)
    
    Returns:
        Lista di nomi delle colonne
    """
    if pd.isna(rem_col_str) or not isinstance(rem_col_str, str) or rem_col_str.strip() == '':
        return []
    
    rem_col_str = rem_col_str.strip()
    
    # Prova a interpretare come lista Python
    if rem_col_str.startswith('[') and rem_col_str.endswith(']'):
        try:
            columns = ast.literal_eval(rem_col_str)
            return columns if isinstance(columns, list) else []
        except (ValueError, SyntaxError):
            logger.warning(f"Impossibile parsare rem_col come lista Python: {rem_col_str}")
            return []
    
    # Altrimenti split per virgola
    columns = [col.strip() for col in rem_col_str.split(',')]
    return [col for col in columns if col]

def load_dataset_with_removal(csv_path: str, exec_type: str, rem_columns: list) -> pd.DataFrame:
    """
    Carica il dataset dal CSV applicando le modifiche secondo exec_type.
    
    Args:
        csv_path: path al file CSV del dataset
        exec_type: modalità di esecuzione ("NORMAL", "NULL", "REMOVE")
        rem_columns: lista di colonne da rimuovere/nullificare
    
    Returns:
        DataFrame modificato
    """
    try:
        df = pd.read_csv(csv_path)
        
        if exec_type.upper() == "REMOVE":
            for col in rem_columns:
                if col in df.columns:
                    df = df.drop(columns=[col], errors='ignore')
        
        elif exec_type.upper() == "NULL":
            for col in rem_columns:
                if col in df.columns:
                    df[col] = None
        
        return df
    
    except Exception as e:
        logger.error(f"Errore caricamento dataset {csv_path}: {e}")
        return None

def format_dataset_for_prompt(df: pd.DataFrame) -> str:
    """
    Formatta il dataset per il prompt (come JSON).
    
    Args:
        df: DataFrame del dataset
    
    Returns:
        Dataset formattato come stringa JSON
    """
    return df.to_json(orient='records', indent=2)

def create_prompt(dataset_str: str, nl_query: str) -> str:
    """
    Crea il prompt completo per l'API.
    
    Args:
        dataset_str: dataset formattato come stringa
        nl_query: query in linguaggio naturale
    
    Returns:
        Prompt completo
    """
    prompt = f"""You are a movie recommendation system. Analyze the provided movie dataset and respond to the user's request.

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

def count_tokens(text: str, model: str = "gpt-4") -> int:
    """
    Conta i token utilizzando tiktoken.
    
    Args:
        text: testo da analizzare
        model: modello per il tokenizer
    
    Returns:
        Numero di token
    """
    try:
        model_encoding_map = {
            "gpt-4": "cl100k_base",
            "gpt-4-turbo": "cl100k_base", 
            "gpt-3.5-turbo": "cl100k_base",
            "gemini": "cl100k_base",  # Usa encoding simile per Gemini
        }
        
        # Normalizza il nome del modello
        model_lower = model.lower()
        encoding_name = "cl100k_base"
        
        for key in model_encoding_map:
            if key in model_lower:
                encoding_name = model_encoding_map[key]
                break
        
        encoding = tiktoken.get_encoding(encoding_name)
        tokens = encoding.encode(text)
        return len(tokens)
    
    except Exception as e:
        logger.error(f"Errore conteggio token: {e}")
        return len(text) // 4

def analyze_query_complexity(query: str) -> dict:
    """
    Analizza la complessità lessicale della query in linguaggio naturale.
    
    Args:
        query: query in linguaggio naturale
    
    Returns:
        Dizionario con metriche di complessità
    """
    if pd.isna(query) or not isinstance(query, str):
        return {
            'query_length': 0,
            'query_word_count': 0,
            'query_avg_word_length': 0,
            'query_unique_words': 0,
            'query_lexical_diversity': 0
        }
    
    # Lunghezza totale
    query_length = len(query)
    
    # Conta parole
    words = query.split()
    word_count = len(words)
    
    # Lunghezza media parole
    avg_word_length = np.mean([len(word) for word in words]) if words else 0
    
    # Parole uniche
    unique_words = len(set(word.lower() for word in words if word.isalpha()))
    
    # Diversità lessicale (rapporto parole uniche / totali)
    lexical_diversity = unique_words / word_count if word_count > 0 else 0
    
    return {
        'query_length': query_length,
        'query_word_count': word_count,
        'query_avg_word_length': round(avg_word_length, 2),
        'query_unique_words': unique_words,
        'query_lexical_diversity': round(lexical_diversity, 3)
    }

def enrich_row(row: pd.Series) -> dict:
    """
    Arricchisce una singola riga con le metriche.
    
    Args:
        row: riga del DataFrame
    
    Returns:
        Dizionario con le nuove metriche
    """
    try:
        # Estrai colonne rimosse
        rem_columns = parse_removed_columns(row.get('rem_col', ''))
        num_removed_columns = len(rem_columns)
        
        # Carica dataset
        csv_path = row.get('csv_path', '')
        exec_type = row.get('exec_type', 'NORMAL')
        
        # Gestisci exec_type NULL o vuoto
        if pd.isna(exec_type) or exec_type == 'NULL' or exec_type == '':
            exec_type = 'NORMAL'
        
        if not csv_path or pd.isna(csv_path) or not os.path.exists(csv_path):
            logger.warning(f"Dataset non trovato: {csv_path}")
            return {
                'dataset_num_rows': None,
                'dataset_num_columns_original': None,
                'dataset_num_columns_after_removal': None,
                'num_removed_columns': num_removed_columns,
                'query_length': None,
                'query_word_count': None,
                'query_avg_word_length': None,
                'query_unique_words': None,
                'query_lexical_diversity': None,
                'token_prompt_count': None
            }
        
        # Carica dataset originale per contare colonne
        df_original = pd.read_csv(csv_path)
        num_rows = len(df_original)
        num_columns_original = len(df_original.columns)
        
        # Carica dataset con rimozioni applicate
        df_modified = load_dataset_with_removal(csv_path, exec_type, rem_columns)
        
        if df_modified is None:
            num_columns_after = num_columns_original
        else:
            num_columns_after = len(df_modified.columns)
        
        # Analizza complessità query
        nl_query = row.get('nl_query', '')
        query_metrics = analyze_query_complexity(nl_query)
        
        # Calcola token count
        token_count = 0
        if df_modified is not None:
            try:
                dataset_str = format_dataset_for_prompt(df_modified)
                full_prompt = create_prompt(dataset_str, str(nl_query))
                model = row.get('model', 'gpt-4')
                token_count = count_tokens(full_prompt, model)
            except Exception as e:
                logger.error(f"Errore calcolo token: {e}")
                token_count = 0
        
        return {
            'dataset_num_rows': num_rows,
            'dataset_num_columns_original': num_columns_original,
            'dataset_num_columns_after_removal': num_columns_after,
            'num_removed_columns': num_removed_columns,
            **query_metrics,
            'token_prompt_count': token_count
        }
    
    except Exception as e:
        logger.error(f"Errore elaborazione riga: {e}")
        return {
            'dataset_num_rows': None,
            'dataset_num_columns_original': None,
            'dataset_num_columns_after_removal': None,
            'num_removed_columns': None,
            'query_length': None,
            'query_word_count': None,
            'query_avg_word_length': None,
            'query_unique_words': None,
            'query_lexical_diversity': None,
            'token_prompt_count': None
        }

def main(input_csv: str, output_csv: str):
    """
    Funzione principale per arricchire il CSV con le metriche.
    
    Args:
        input_csv: path del file CSV di input
        output_csv: path del file CSV di output
    """
    try:
        # Verifica tiktoken
        try:
            import tiktoken
        except ImportError:
            logger.error("tiktoken non installato. Installa con: pip install tiktoken")
            raise
        
        # Carica CSV
        print("📂 Caricamento CSV...")
        df = pd.read_csv(input_csv)
        logger.info(f"Caricato CSV con {len(df)} righe")
        
        # Mostra colonne disponibili
        print(f"📋 Colonne disponibili: {list(df.columns)}")
        
        # Verifica colonne necessarie
        required_cols = ["csv_path", "nl_query"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Colonne mancanti: {missing_cols}")
        
        # Rimuovi colonna duplicata se presente
        if 'num_tests.1' in df.columns:
            df = df.drop(columns=['num_tests.1'])
            print("🗑️  Rimossa colonna duplicata 'num_tests.1'")
        
        # Lista per le nuove feature
        new_features = []
        errors = 0
        
        # Progress bar
        print(f"🚀 Elaborazione di {len(df)} righe...")
        with tqdm(total=len(df), desc="Processando", unit="righe") as pbar:
            for i, row in df.iterrows():
                pbar.set_postfix_str(f"Riga {i+1} | Errori: {errors}")
                
                features = enrich_row(row)
                
                if features['token_prompt_count'] is None:
                    errors += 1
                
                new_features.append(features)
                pbar.update(1)
        
        # Crea DataFrame delle feature
        print("📊 Creazione DataFrame finale...")
        features_df = pd.DataFrame(new_features)
        
        # Unisci al dataset originale
        result = pd.concat([df, features_df], axis=1)
        
        # Crea directory di output
        os.makedirs(os.path.dirname(output_csv) if os.path.dirname(output_csv) else '.', exist_ok=True)
        
        # Salva risultato
        print("💾 Salvataggio risultati...")
        result.to_csv(output_csv, index=False)
        
        # Report finale
        print("\n" + "="*60)
        print("✅ ELABORAZIONE COMPLETATA!")
        print("="*60)
        print(f"📁 File input: {input_csv}")
        print(f"💾 File output: {output_csv}")
        print(f"📊 Righe elaborate: {len(df)}")
        print(f"🔢 Feature aggiunte: {len(features_df.columns)}")
        print(f"⚠️  Errori: {errors}/{len(df)} ({errors/len(df)*100:.1f}%)")
        
        # Statistiche sui token
        valid_tokens = result['token_prompt_count'].dropna()
        if len(valid_tokens) > 0:
            print(f"\n🔤 Statistiche Token:")
            print(f"   Token totali: {valid_tokens.sum():,.0f}")
            print(f"   Token medi: {valid_tokens.mean():.2f}")
            print(f"   Token minimi: {valid_tokens.min():,.0f}")
            print(f"   Token massimi: {valid_tokens.max():,.0f}")
        
        # Statistiche dataset
        print(f"\n📊 Statistiche Dataset:")
        if 'dataset_num_rows' in result.columns:
            rows_stats = result['dataset_num_rows'].describe()
            print(f"   Righe medie: {rows_stats['mean']:.0f}")
            print(f"   Righe min/max: {rows_stats['min']:.0f} / {rows_stats['max']:.0f}")
        
        if 'num_removed_columns' in result.columns:
            removed = result['num_removed_columns'].value_counts().sort_index()
            print(f"\n🗑️  Distribuzione colonne rimosse:")
            for num, count in removed.items():
                print(f"   {num} colonne: {count} righe")
        
        # Statistiche query
        print(f"\n📝 Statistiche Query:")
        if 'query_word_count' in result.columns:
            words = result['query_word_count'].describe()
            print(f"   Parole medie: {words['mean']:.1f}")
            print(f"   Parole min/max: {words['min']:.0f} / {words['max']:.0f}")
        
        if 'query_lexical_diversity' in result.columns:
            diversity = result['query_lexical_diversity'].mean()
            print(f"   Diversità lessicale media: {diversity:.3f}")
        
    except Exception as e:
        logger.error(f"❌ Errore fatale: {e}")
        raise

if __name__ == "__main__":
    import sys
    
    input_csv = sys.argv[1] if len(sys.argv) > 1 else "results/final_unified_results.csv"
    output_csv = sys.argv[2] if len(sys.argv) > 2 else "enrich_res/enriched_final_results.csv"
    
    main(input_csv, output_csv)