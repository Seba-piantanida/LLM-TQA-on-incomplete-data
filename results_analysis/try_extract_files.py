import os

# --- IMPOSTAZIONI ---
cartella_principale = 'raw_results'  # Cambia con il percorso della tua cartella principale

# Lista vuota per memorizzare i percorsi dei file CSV trovati
lista_file_csv = []

print("Ricerca in corso...")
for cartella, sottocartelle, files in os.walk(cartella_principale):
    for nome_file in files:
        # 👇 CONDIZIONE AGGIORNATA 👇
        # Cerca i file che finiscono con '.csv' E che NON iniziano con '._'
        if nome_file.endswith('.csv') and not nome_file.startswith('._'):
            # Costruisce il percorso completo del file e lo aggiunge alla lista
            percorso_completo = os.path.join(cartella, nome_file)
            lista_file_csv.append(percorso_completo)

# --- OUTPUT ---
print("\n" + "="*50)
print("✅ Elenco dei file CSV trovati (ignorando i file '._')")
print("="*50)

if lista_file_csv:
    for file in lista_file_csv:
        print(file)
else:
    print("Nessun file CSV valido è stato trovato.")

print("\n" + "="*50)
print(f"📄 Numero totale di file CSV validi: {len(lista_file_csv)}")
print("="*50)