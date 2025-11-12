
import pandas as pd

# Leggi il CSV
df = pd.read_csv('variance_test/variance_tests_QATCH.csv')
dim_iniziale = df.shape[0]
# Filtra le righe dove AI_answer è una lista vuota (cioè '[]' come stringa)
df = df[df['AI_answer'] != '[]']
dim_finale = df.shape[0]

perc_fail = dim_finale/dim_iniziale

print(f"succ rate = {perc_fail}%")

# Specifica le colonne su cui calcolare la varianza
colonne_metriche = [
    'cell_precision', 'cell_recall', 'execution_accuracy',
    'tuple_cardinality', 'tuple_constraint', 'tuple_order'
]

# Raggruppa per 'model' e 'query', e calcola la varianza delle metriche
varianze = df.groupby(['model', 'query'])[colonne_metriche].var()

# Opzionalmente, salva il risultato su CSV
varianze.to_csv('variance_test/variance_group.csv')

varianze_medie = df.groupby(['model'])[colonne_metriche].var()
print(varianze_medie)

