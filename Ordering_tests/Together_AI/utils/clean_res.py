import pandas as pd
import google.generativeai as genai
import os
import re
import time
import ast
from tqdm import tqdm

def get_gemini_response(text_to_fix, model):
    """
    Invia il testo malformato a Gemini e chiede di convertirlo in una lista di liste.
    """
    prompt = f"""
    Analizza il seguente testo, che è uno snippet JSON incompleto e potenzialmente malformato estratto da un file di log.
    Il tuo unico compito è estrarre i dati dalla chiave "ordered_entries" e convertirli in una lista Python di liste.
    Ogni lista interna deve contenere solo i valori dell'oggetto JSON, mantenendo il loro ordine originale.

    REGOLE IMPORTANTI:
    1. Il tuo output deve essere ESCLUSIVAMENTE la lista di liste Python, formattata come una stringa.
    2. NON includere spiegazioni, commenti, o la parola "python" o ```.
    3. Se l'input è vuoto o non contiene dati validi, restituisci una lista vuota: [].

    --- ESEMPIO ---
    INPUT: {{"table_name": "movies", "ordered_entries": [ {{"Title": "Titanic", "Year": 1997}}, {{"Title": "Avatar", "Year": 2009}} ]}}'
    OUTPUT CORRETTO: '[['Titanic', 1997], ['Avatar', 2009]]'
    --- FINE ESEMPIO ---

    Ora elabora il seguente input:
    '{text_to_fix}'
    """
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"  -> Errore durante la chiamata API: {e}")
        return None

def process_csv_with_gemini(input_filepath, output_filepath):
    """
    Scansiona un CSV, usa Gemini per correggere le righe problematiche e salva un nuovo file.
    """
    # 1. Configurazione API
    try:
        api_key = 'AIzaSyChjGg9UP0HQPYyitR5wcWIIGjp368bXMA'
        if not api_key:
            print("❌ ERRORE: La variabile d'ambiente GOOGLE_API_KEY non è stata impostata.")
            return
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash-lite')
    except Exception as e:
        print(f"❌ ERRORE durante la configurazione dell'API: {e}")
        return

    # 2. Caricamento Dati
    try:
        df = pd.read_csv(input_filepath, dtype=str).fillna('')
    except FileNotFoundError:
        print(f"❌ ERRORE: Il file '{input_filepath}' non è stato trovato.")
        return

    print(f"✅ File '{input_filepath}' caricato. Righe totali: {len(df)}")
    print("🤖 Avvio elaborazione con Gemini. Potrebbe richiedere tempo...")

    righe_modificate = 0
    # Usiamo tqdm per una barra di avanzamento
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Elaborazione righe"):
        note = row['note']
        ai_answer = row['AI_answer']

        # 3. Condizione per l'elaborazione
        if note and '"ordered_entries"' in note and (not ai_answer or ai_answer.strip() == '[]'):
            # Estrae solo la parte che sembra un JSON per non sprecare token
            start_index = note.find('{"table_name":')
            text_to_process = note[start_index:] if start_index != -1 else note

            # 4. Chiamata API
            gemini_result = get_gemini_response(text_to_process, model)
            
            if gemini_result:
                # 5. Validazione e aggiornamento del DataFrame
                try:
                    # ast.literal_eval è un modo sicuro per verificare se la stringa
                    # è una struttura dati Python valida (come una lista di liste).
                    parsed_list = ast.literal_eval(gemini_result)
                    if isinstance(parsed_list, list):
                        df.at[index, 'AI_answer'] = gemini_result
                        df.at[index, 'note'] = '' # Pulisce la colonna 'note'
                        righe_modificate += 1
                    else:
                        print(f"\nAVVISO Riga {index + 2}: Gemini ha restituito un tipo non valido: {type(parsed_list)}")
                except (ValueError, SyntaxError):
                    print(f"\nAVVISO Riga {index + 2}: Gemini ha restituito una stringa malformata: {gemini_result}")
            
            # Pausa per non superare i limiti di chiamate API (60 al minuto per Flash)
            time.sleep(1.1)

    # 6. Salvataggio del file finale
    try:
        df.to_csv(output_filepath, index=False)
        print("\n" + "="*50)
        print(f"🎉 Operazione completata! Righe modificate: {righe_modificate}")
        print(f"Il nuovo file è stato salvato in: '{output_filepath}'")
        print("="*50)
    except Exception as e:
        print(f"❌ ERRORE: Impossibile salvare il file di output. Dettaglio: {e}")

# --- Esecuzione dello script ---
if __name__ == "__main__":
    file_input = 'output/categorical_simple_eng/out_categorical_simple_eng_normal_QATCH.csv'
    file_output = 'output/categorical_simple_eng/out_categorical_simple_eng_normal_QATCH_GEMINI.csv'
    
    process_csv_with_gemini(file_input, file_output)