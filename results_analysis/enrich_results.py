import pandas as pd
import sqlite3
import re
import os
import numpy as np
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import logging
from typing import Dict, List, Optional, Tuple
import warnings
from tqdm import tqdm
import requests
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoTokenizer

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore")

# Disabilita progress bar di librerie esterne
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Lazy loading dei modelli NLP
nlp = None
embed_model = None
pca = None
scaler = StandardScaler()
tokenizer = None

def load_models():
    """Carica i modelli NLP solo quando necessario - SENZA progress bar"""
    global nlp, embed_model
    
    if nlp is None:
        try:
            # Disabilita output di spaCy
            import sys
            original_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')
            nlp = spacy.load("en_core_web_sm")
            sys.stdout = original_stdout
        except OSError:
            logger.warning("Modello spaCy non trovato. Installare con: python -m spacy download en_core_web_sm")
            nlp = False
    
    if embed_model is None:
        try:
            # Disabilita progress bar di SentenceTransformer
            device = "cuda" if torch.cuda.is_available() else "cpu"
            embed_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
        except Exception as e:
            logger.warning(f"Errore caricamento SentenceTransformer: {e}")
            embed_model = False
    
    if tokenizer is None:
        try:
            # Carica tokenizer una volta sola
            tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        except Exception as e:
            logger.warning(f"Errore caricamento tokenizer: {e}")
            tokenizer = False

AGG_FUNCTIONS = {"sum", "avg", "average", "max", "min", "count", "total", "mean", "median"}
COMPLEXITY_KEYWORDS = [
    "group", "order", "sort", "sum", "average", "mean", "filter", "where", "having",
    "top", "limit", "most", "least", "more than", "less than", "greater", "smaller",
    "compared to", "difference", "total", "change", "per", "each", "across",
    "join", "inner", "outer", "left", "right", "union", "intersect", "except",
    "distinct", "unique", "duplicate", "aggregate", "partition"
]

def is_numeric_type(sql_type: str) -> bool:
    """Verifica se un tipo SQL è numerico"""
    if sql_type is None:
        return False
    numeric_keywords = ["int", "real", "double", "float", "decimal", "numeric", "number"]
    return any(kw in sql_type.lower() for kw in numeric_keywords)

def safe_db_operation(func):
    """Decorator per operazioni DB sicure"""
    def wrapper(*args, **kwargs):
        conn = None
        try:
            return func(*args, **kwargs)
        except sqlite3.Error as e:
            logger.error(f"Errore database: {e}")
            return None
        except Exception as e:
            logger.error(f"Errore generico in {func.__name__}: {e}")
            return None
        finally:
            if conn:
                conn.close()
    return wrapper

def parse_table_names(table_name_str: str) -> List[str]:
    """Estrae i nomi delle tabelle da una stringa che può contenere più tabelle separate da virgole"""
    if pd.isna(table_name_str) or not isinstance(table_name_str, str):
        return []
    
    # Rimuovi spazi extra e split per virgole
    tables = [table.strip() for table in str(table_name_str).split(',')]
    # Rimuovi stringhe vuote
    tables = [table for table in tables if table]
    
    return tables

@safe_db_operation
def analyze_single_table(db_path: str, table_name: str) -> Optional[Dict]:
    """Analizza una singola tabella del database"""
    if not os.path.exists(db_path):
        logger.error(f"Database non trovato: {db_path}")
        return None
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Info colonne
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()
        
        if not columns:
            logger.warning(f"Tabella {table_name} non trovata o vuota")
            conn.close()
            return None
        
        col_names = [col[1] for col in columns]
        col_types = [col[2] for col in columns]
        
        # Carica dati con limite per performance
        try:
            df = pd.read_sql_query(f"SELECT * FROM {table_name} LIMIT 10000", conn)
        except Exception as e:
            logger.error(f"Errore lettura tabella {table_name}: {e}")
            conn.close()
            return None
        
        conn.close()
        
        if df.empty:
            return {
                "table_name": table_name,
                "num_columns": len(col_names),
                "num_rows": 0,
                "num_numeric_columns": 0,
                "num_categorical_columns": len(col_names),
                "numeric_categorical_ratio": 0,
                "avg_unique_vals": 0,
                "max_unique_vals": 0,
                "data_sparsity": 1.0
            }
        
        # Classificazione colonne migliorata
        numeric_cols = []
        categorical_cols = []
        
        for col_name, col_type in zip(col_names, col_types):
            if col_name in df.columns:
                if is_numeric_type(col_type) or pd.api.types.is_numeric_dtype(df[col_name]):
                    numeric_cols.append(col_name)
                else:
                    categorical_cols.append(col_name)
        
        # Metriche avanzate
        sparsity = (df.isnull().sum() / len(df)).mean()
        avg_cardinality = df.nunique().mean()
        max_cardinality = df.nunique().max()

        normalized_cardinality = (df.nunique() / len(df)).mean()
        
        # Distribuzione tipi di dati
        data_type_distribution = df.dtypes.value_counts(normalize=True).to_dict()
        
        return {
            "table_name": table_name,
            "num_columns": len(col_names),
            "num_rows": len(df),
            "num_numeric_columns": len(numeric_cols),
            "num_categorical_columns": len(categorical_cols),
            "numeric_categorical_ratio": len(numeric_cols) / max(len(categorical_cols), 1),
            "avg_unique_vals": avg_cardinality,
            "max_unique_vals": max_cardinality,
            "normalized_cardinality": normalized_cardinality,
            "data_sparsity": sparsity,
            "columns": col_names,
            "numeric_columns": numeric_cols,
            "categorical_columns": categorical_cols,
            "data_type_distribution": data_type_distribution
        }
        
    except Exception as e:
        logger.error(f"Errore nell'analisi della tabella {table_name}: {e}")
        return None

def aggregate_table_analyses(table_analyses: List[Dict]) -> Dict:
    """Aggrega le analisi di più tabelle in un'unica analisi combinata"""
    if not table_analyses:
        return {
            "num_tables": 0,
            "total_columns": 0,
            "total_rows": 0,
            "total_numeric_columns": 0,
            "total_categorical_columns": 0,
            "avg_numeric_categorical_ratio": 0,
            "avg_unique_vals": 0,
            "max_unique_vals": 0,
            "avg_data_sparsity": 0,
            "table_names": []
        }
    
    # Filtra le analisi valide
    valid_analyses = [analysis for analysis in table_analyses if analysis is not None]
    
    if not valid_analyses:
        return {
            "num_tables": 0,
            "total_columns": 0,
            "total_rows": 0,
            "total_numeric_columns": 0,
            "total_categorical_columns": 0,
            "avg_numeric_categorical_ratio": 0,
            "avg_unique_vals": 0,
            "max_unique_vals": 0,
            "avg_data_sparsity": 0,
            "table_names": []
        }
    
    # Aggrega le metriche
    total_columns = sum(a["num_columns"] for a in valid_analyses)
    total_rows = sum(a["num_rows"] for a in valid_analyses)
    total_numeric_columns = sum(a["num_numeric_columns"] for a in valid_analyses)
    total_categorical_columns = sum(a["num_categorical_columns"] for a in valid_analyses)
    
    # Medie ponderate e aggregate
    avg_numeric_categorical_ratio = np.mean([a["numeric_categorical_ratio"] for a in valid_analyses])
    avg_unique_vals = np.mean([a["avg_unique_vals"] for a in valid_analyses])
    max_unique_vals = max(a["max_unique_vals"] for a in valid_analyses)
    avg_data_sparsity = np.mean([a["data_sparsity"] for a in valid_analyses])
    
    table_names = [a["table_name"] for a in valid_analyses]
    
    return {
        "num_tables": len(valid_analyses),
        "total_columns": total_columns,
        "total_rows": total_rows,
        "total_numeric_columns": total_numeric_columns,
        "total_categorical_columns": total_categorical_columns,
        "avg_columns_per_table": total_columns / len(valid_analyses),
        "avg_rows_per_table": total_rows / len(valid_analyses),
        "avg_numeric_categorical_ratio": avg_numeric_categorical_ratio,
        "avg_unique_vals": avg_unique_vals,
        "max_unique_vals": max_unique_vals,
        "avg_data_sparsity": avg_data_sparsity,
        "table_names": table_names,
        
        "num_columns": total_columns,
        "num_rows": total_rows,
        "num_numeric_columns": total_numeric_columns,
        "num_categorical_columns": total_categorical_columns,
        "numeric_categorical_ratio": avg_numeric_categorical_ratio
    }

def analyze_tables(db_path: str, table_names_str: str) -> Optional[Dict]:
    """Analizza una o più tabelle del database"""
    table_names = parse_table_names(table_names_str)
    
    if not table_names:
        logger.error(f"Nessun nome tabella valido trovato in: {table_names_str}")
        return None
    
    # Analizza ogni tabella singolarmente
    table_analyses = []
    for table_name in table_names:
        analysis = analyze_single_table(db_path, table_name)
        if analysis is not None:
            table_analyses.append(analysis)
        else:
            logger.warning(f"Impossibile analizzare la tabella: {table_name}")
    
    # Aggrega i risultati
    return aggregate_table_analyses(table_analyses)

def analyze_query_advanced(sql_query: str, table_info: Dict) -> Dict:
    """Analisi avanzata della query SQL"""
    if not isinstance(sql_query, str) or pd.isna(sql_query):
        sql = ""   # fallback vuoto se non è stringa
    else:
        sql = sql_query.lower().strip()
    
    # Conta clausole SQL
    clauses = {
        "has_where": "where" in sql,
        "has_group_by": "group by" in sql,
        "has_having": "having" in sql,
        "has_order_by": "order by" in sql,
        "has_limit": "limit" in sql,
        "has_distinct": "distinct" in sql,
        "has_join": any(join_type in sql for join_type in ["join", "inner join", "left join", "right join", "outer join"]),
        "has_subquery": "(" in sql and "select" in sql.split("(")[1] if "(" in sql else False,
        "has_union": "union" in sql,
        "has_aggregation": any(agg in sql for agg in AGG_FUNCTIONS)
    }

    # Estrae il valore del LIMIT se presente
    limit_value = None
    if clauses["has_limit"]:
        # Cerca pattern LIMIT seguito da numero (opzionalmente con OFFSET)
        limit_match = re.search(r"limit\s+(\d+)(?:\s+offset\s+\d+)?", sql, re.IGNORECASE)
        if not limit_match:
            # Cerca anche il pattern LIMIT offset, number (MySQL style)
            limit_match = re.search(r"limit\s+\d+\s*,\s*(\d+)", sql, re.IGNORECASE)
        
        if limit_match:
            try:
                limit_value = int(limit_match.group(1))
            except (ValueError, IndexError):
                limit_value = None
    
    # Conta colonne selezionate (usa total_columns invece di num_columns per compatibilità)
    total_columns = table_info.get("total_columns", table_info.get("num_columns", 0))
    
    if "select *" in sql:
        num_selected = total_columns
    else:
        select_match = re.search(r"select\s+(.*?)\s+from", sql, re.DOTALL | re.IGNORECASE)
        if select_match:
            selected = select_match.group(1)
            # Rimuovi funzioni aggregate per contare colonne reali
            selected_clean = re.sub(r'\w+\([^)]+\)', '', selected)
            num_selected = len([s.strip() for s in selected_clean.split(",") if s.strip()])
        else:
            num_selected = 0
    
    # Analizza ORDER BY
    order_cols = []
    order_match = re.search(r"order\s+by\s+(.*?)(?:limit|$)", sql, re.IGNORECASE | re.DOTALL)
    if order_match:
        order_expr = order_match.group(1)

        # Prendi colonne tra apici ('colonna con spazi') o identificatori semplici
        order_cols = re.findall(r"'([^']+)'|\"([^\"]+)\"|(\w+)", order_expr)

        # Flatten risultati e togli None
        order_cols = [col for group in order_cols for col in group if col]

        # Filtra asc/desc
        order_cols = [col for col in order_cols if col.lower() not in ("asc", "desc")]

    # Complessità sintattica
    active_clauses = sum(1 for v in clauses.values() if v)

    syntax_complexity = (
        active_clauses
        + num_selected * 0.3  # selezionare più colonne aumenta ma poco
        + len(order_cols) * 0.5
        + clauses["has_join"] * 2.0
        + clauses["has_subquery"] * 3.0
        + clauses["has_union"] * 2.5
        + (sql.lower().count(" and ") + sql.lower().count(" or ")) * 0.5
    )

    return {
        "num_selected_columns": num_selected,
        "num_order_columns": len(order_cols),
        "limit_value": limit_value,
        "syntax_complexity": syntax_complexity,
        **clauses
    }

def query_complexity_score_advanced(text: str, df: pd.DataFrame) -> Dict:
    """Calcolo avanzato della complessità della query - SENZA progress bar interne"""
    if not nlp or not embed_model:
        # Fallback senza NLP
        return {
            
            "text_length": len(text),
            "num_tokens": len(text.split()),
            "num_words": len([w for w in text.split() if w.isalpha()]),
            "keyword_score": sum(kw in text.lower() for kw in COMPLEXITY_KEYWORDS[:10]),
            "semantic_complexity": 0
        }
    
    try:
        doc = nlp(text)
        
        # Feature linguistiche dettagliate
        global tokenizer
        try:
            if tokenizer:
                tokens = tokenizer.tokenize(text)
                num_tokens = len(tokens)
            else:
                num_tokens = len(text.split())
        except Exception as e:
            num_tokens = len(text.split())

        num_words = len([t for t in doc if t.is_alpha])
        num_nouns = len([t for t in doc if t.pos_ == "NOUN"])
        num_verbs = len([t for t in doc if t.pos_ == "VERB"])
        num_adjectives = len([t for t in doc if t.pos_ == "ADJ"])
        num_entities = len(doc.ents)
        
        # Complessità sintattica
        num_clauses = len([t for t in doc if t.dep_ in ("advcl", "relcl", "ccomp", "xcomp")])
        avg_sentence_length = np.mean([len(sent) for sent in doc.sents]) if list(doc.sents) else 0
        
        # Keyword semantiche
        keyword_score = sum(kw in text.lower() for kw in COMPLEXITY_KEYWORDS)
        
        # Complessità semantica con embedding - SENZA progress bar
        semantic_complexity = 0
        try:
            if len(df) > 1:
                all_texts = df["query"].dropna().tolist()
                if len(all_texts) > 1:
                    all_embeddings = embed_model.encode(all_texts[:2000], show_progress_bar=False, normalize_embeddings=True)

                    # centroide del dataset
                    centroid = np.mean(all_embeddings, axis=0)

                    # embedding query
                    text_embedding = embed_model.encode([text], show_progress_bar=False, normalize_embeddings=True)[0]

                    # distanza coseno dal centroide (più distante = più "complessa/atipica")
                    semantic_complexity = float(1 - np.dot(text_embedding, centroid))
        except Exception as e:
            logger.warning(f"Errore calcolo complessità semantica: {e}")
            semantic_complexity = 0
                
     
        
        return {
            
            "text_length": len(text),
            "num_tokens": num_tokens,
            "num_words": num_words,
            "num_nouns": num_nouns,
            "num_verbs": num_verbs,
            "num_adjectives": num_adjectives,
            "num_entities": num_entities,
            "num_clauses": num_clauses,
            "avg_sentence_length": avg_sentence_length,
            "keyword_score": keyword_score,
            "semantic_complexity": semantic_complexity
        }
        
    except Exception as e:
        logger.error(f"Errore analisi NLP: {e}")
        # Fallback semplificato
        return {
            "text_length": len(text),
            "num_tokens": len(text.split()),
            "keyword_score": sum(kw in text.lower() for kw in COMPLEXITY_KEYWORDS[:5]),
            "semantic_complexity": 0
        }

def main(input_csv: str, output_csv: str):
    """Funzione principale con UNICA progress bar globale e supporto per tabelle multiple"""
    try:
        # Carica CSV
        print("📂 Caricamento CSV...")
        df = pd.read_csv(input_csv)
        logger.info(f"Caricato CSV con {len(df)} righe")
        
        # Verifica colonne necessarie
        required_cols = ["db_path", "table_name", "SQL_query", "query"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Colonne mancanti: {missing_cols}")
        
        # Carica modelli una sola volta all'inizio
        print("🤖 Caricamento modelli NLP...")
        load_models()
        print("✅ Modelli caricati")
        
        new_features = []
        errors = 0
        multi_table_count = 0
        
        # 🎯 UNICA PROGRESS BAR GLOBALE
        print(f"🚀 Elaborazione di {len(df)} righe...")
        with tqdm(total=len(df), desc="Processando", unit="righe", 
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}") as pbar:
            
            for i, row in df.iterrows():
                try:
                    # Controlla se ci sono tabelle multiple
                    table_names = parse_table_names(row["table_name"])
                    if len(table_names) > 1:
                        multi_table_count += 1
                    
                    # Update della descrizione con info corrente
                    pbar.set_postfix_str(f"Riga {i+1} | Tabelle: {len(table_names)} | Errori: {errors}")
                    
                    # Analisi tabelle (ora supporta tabelle multiple)
                    table_info = analyze_tables(row["db_path"], row["table_name"])
                    if table_info is None:
                        raise ValueError("Impossibile analizzare le tabelle")
                    
                    # Analisi query SQL
                    query_info = analyze_query_advanced(row["SQL_query"], table_info)
                    
                    # Analisi complessità linguistica
                    complexity_info = query_complexity_score_advanced(str(row["query"]), df)
                    
                    # Combina tutte le feature
                    combined = {**table_info, **query_info, **complexity_info}
                    
                    # Rimuovi liste e dizionari che non possono essere serializzati in CSV
                    combined = {k: v for k, v in combined.items() 
                               if not isinstance(v, (list, dict))}
                    
                except Exception as e:
                    logger.error(f"Errore alla riga {i}: {e}")
                    errors += 1
                    
                    # Feature di default in caso di errore
                    combined = {
                        "num_tables": None, "total_columns": None, "total_rows": None, 
                        "total_numeric_columns": None, "total_categorical_columns": None,
                        "avg_columns_per_table": None, "avg_rows_per_table": None,
                        "avg_numeric_categorical_ratio": None, "avg_unique_vals": None, 
                        "max_unique_vals": None, "avg_data_sparsity": None,
                        "num_columns": None, "num_rows": None, "num_numeric_columns": None,
                        "num_categorical_columns": None, "numeric_categorical_ratio": None,
                        "num_selected_columns": None, "num_order_columns": None,
                        "syntax_complexity": None, "query_complexity": None,
                        "text_length": None, "num_tokens": None, "keyword_score": None,
                        "semantic_complexity": None
                    }
                
                new_features.append(combined)
                
                # Aggiorna progress bar
                pbar.update(1)
        
        # Crea DataFrame delle feature e uniscilo al dataset originale
        print("📊 Creazione DataFrame finale...")
        features_df = pd.DataFrame(new_features)
        result = pd.concat([df, features_df], axis=1)
        
        # Crea directory di output se non exists
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
        print(f"📋 Righe con tabelle multiple: {multi_table_count}")
        print(f"⚠️  Errori: {errors}/{len(df)} ({errors/len(df)*100:.1f}%)")
        
        # Statistiche descrittive delle nuove feature
        numeric_features = features_df.select_dtypes(include=[np.number])
        if not numeric_features.empty:
            print("\n📈 Statistiche feature numeriche:")
            print(numeric_features.describe())
        
        # Statistiche sulle tabelle multiple
        if multi_table_count > 0:
            print(f"\n🔗 Statistiche tabelle multiple:")
            print(f"   Righe con tabelle multiple: {multi_table_count}")
            print(f"   Percentuale: {multi_table_count/len(df)*100:.1f}%")
            
            # Analizza la distribuzione del numero di tabelle
            num_tables_series = features_df['num_tables'].dropna()
            if not num_tables_series.empty:
                print(f"   Media tabelle per riga: {num_tables_series.mean():.2f}")
                print(f"   Max tabelle per riga: {num_tables_series.max()}")
        
    except Exception as e:
        logger.error(f"❌ Errore fatale: {e}")
        raise

if __name__ == "__main__":
    import sys
    
    input_csv = sys.argv[1] if len(sys.argv) > 1 else "results_all_filtered/normal.csv"
    output_csv = sys.argv[2] if len(sys.argv) > 2 else f"all_res_enrich/{input_csv.replace('.csv', '').replace('results_all_filtered/', '')}_enriched.csv"
    
    main(input_csv, output_csv)
    try:
        load_dotenv()  
        requests.post(
            f"{os.getenv('NTFY_SERVER')}",
            data=f"result enrichment completed",
            auth=(f"{os.getenv('NTFY_USR')}", f"{os.getenv('NTFY_PW')}")
        )
        
    except Exception as e:
        print('error sending notification:', e)