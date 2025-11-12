import random
from itertools import combinations

def get_random_combinations(columns, n, k=None):
    if k is None:
        k = len(columns)
    all_combinations = []
    for r in range(2, k + 1):
        all_combinations.extend(combinations(columns, r))
    random.shuffle(all_combinations)
    n = min(n, len(all_combinations))
    selected_combinations = all_combinations[:n]

    return selected_combinations

# Esempio di utilizzo
columns = ['A', 'B', 'C', 'D']
random_combos = get_random_combinations(columns, n=5, k=len(columns))

for combo in random_combos:
    print(combo)