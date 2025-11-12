# Movie Similarity Bilingual Test Generator

Generatore di test cases per la similarità di film bilingue (Italiano/Inglese) a partire da un dataset IMDB. Include query positive e negative per valutare le performance di modelli LLM nella ricerca semantica.

---

## 📦 Installazione

1. Assicurati di avere **Python 3.8+** installato.
2. Installa i pacchetti richiesti dal file `requirements.txt`:

```bash
pip install -r requirements.txt
```

3. Configura le API keys necessarie:
   - **Google Gemini**: Imposta la variabile d'ambiente `GOOGLE_API_KEY`
   - **OpenAI GPT**: Imposta la variabile d'ambiente `OPENAI_API_KEY`

---

## 🎬 Generazione dei Test Cases

Utilizza lo script `gen_tests_similarity.py` per generare test cases bilingue dal dataset IMDB.

### Sintassi

```bash
python gen_tests_similarity.py <csv_filename> [--max_tests MAX_TESTS] [--tests_per_category TESTS_PER_CATEGORY]
```

### Parametri

- `csv_filename` **(obbligatorio)**: Percorso del file CSV di input contenente i dati IMDB
- `--max_tests`: Numero massimo totale di test da generare (default: 200)
- `--tests_per_category`: Numero di test per categoria (default: max_tests // 4)

### Esempio

```bash
python gen_tests_similarity.py data/imdb_movies.csv --max_tests 300
```

Questo comando genera 300 test cases distribuiti equamente tra le categorie disponibili.

---

## 🤖 Esecuzione Test con Google Gemini

Utilizza lo script `gemini_run.py` per eseguire i test con le API di Google Gemini.

### Sintassi

```bash
python gemini_run.py --test_csv <path_test> --output_dir <output_dir> [opzioni]
```

### Parametri

- `--test_csv` **(obbligatorio)**: Path al file CSV dei test
- `--output_dir`: Directory di output per i risultati (default: `results`)
- `--modes`: Modalità di esecuzione da testare (default: `NORMAL`)
  - `NORMAL`: Esecuzione con tutti i dati
  - `NULL`: Nullificazione delle colonne specificate
  - `REMOVE`: Rimozione delle colonne specificate
- `--rem_columns`: Colonne da rimuovere/nullificare (default: `title year`)
- `--analyze`: Analizza i risultati esistenti invece di eseguire nuovi test
- `--results_file`: Path al file dei risultati da analizzare (usato con `--analyze`)

### Esempi

**Esecuzione test standard:**
```bash
python gemini_run.py --test_csv tests/test_movie_similarity_bilingual_cut_300.csv --output_dir results/gemini
```

**Test con modalità NULL e REMOVE:**
```bash
python gemini_run.py --test_csv tests/test_movie_similarity_bilingual_cut_300.csv \
    --modes NULL REMOVE --rem_columns title year --output_dir results/gemini
```

**Analisi risultati esistenti:**
```bash
python gemini_run.py --analyze --results_file results/gemini/results.csv
```

---

## 🔧 Esecuzione Test con GPT

Per eseguire i test con GPT, utilizza lo script `GPT_run.py` dopo aver configurato il file `tests_gpt.json`.

### Configurazione

1. Modifica il file `tests_gpt.json` con la seguente struttura:

```json
[
    {
        "test_path": "tests/test_movie_similarity_bilingual_cut_300.csv",
        "out_path": "results/GPT/no_all/cut_300",
        "out_name": "cut_300.csv",
        "modes": ["NULL", "REMOVE"]
    },
    {
        "test_path": "tests/test_movie_similarity_bilingual_cut_1000.csv",
        "out_path": "results/GPT/no_all/cut_1000",
        "out_name": "cut_1000.csv",
        "modes": ["NULL", "REMOVE"]
    }
]
```
2. aggiorna al interno dello script i path per 'chrome_path' e 'chrome_profile'
3. appena si avvia il browser effettua l'accesso se necessario a chat GPT e poi premi ENTER nel terminale

### Campi del JSON

- `test_path`: Percorso del file CSV di test
- `out_path`: Directory dove salvare i risultati
- `out_name`: Nome del file di output
- `modes`: Array di modalità da testare (`NORMAL`, `NULL`, `REMOVE`)

### Esecuzione

```bash
python GPT_run.py
```

Lo script leggerà automaticamente la configurazione da `tests_gpt.json` ed eseguirà tutti i test specificati.

---


## 📝 Note

- crea un file .env e inserisci le tue API key:
    GEMINI_API_KEY = 'AIxxxxxxxxx'


--