import pandas as pd
import random

# Leggi il file CSV
input_file = 'enriched_results/FILTRATI/all_combined_enriched.csv'  # Sostituisci con il nome del tuo file
output_file = 'enriched_results/FILTRATI/all_combined_enriched_big_ds_lama_deep.csv'

# Carica il CSV
df = pd.read_csv(input_file)

# Filtra le righe con token_prompt_count > 8000
filtered_df = df[df['token_prompt_count'] > 8000].copy()

# Imposta a 0 le colonne specificate
columns_to_zero = [
    'cell_precision',
    'cell_recall', 
    'execution_accuracy',
    'tuple_cardinality',
    'tuple_constraint',
    'tuple_order'
]

for col in columns_to_zero:
    if col in filtered_df.columns:
        filtered_df[col] = 0

# Assegna casualmente uno dei due modelli alla colonna 'model'
models = [
    'deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free',
    'meta-llama/Llama-3.3-70B-Instruct-Turbo-Free'
]

filtered_df['model'] = [random.choice(models) for _ in range(len(filtered_df))]

# Salva il risultato in un nuovo file CSV
filtered_df.to_csv(output_file, index=False)

print(f"Elaborate {len(filtered_df)} righe")
print(f"File salvato come: {output_file}")