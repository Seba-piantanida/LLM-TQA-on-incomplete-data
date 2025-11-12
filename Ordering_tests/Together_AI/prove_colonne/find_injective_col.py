import sqlite3
import pandas as pd
from itertools import combinations

def find_injective_dependencies(db_path, max_combination_size=None):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    injective_dependencies = {}

    # Troviamo tutte le tabelle
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]

    for table in tables:
        print(f"\nAnalizzo tabella: {table}")
        df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
        columns = df.columns.tolist()
        injective_cols_in_table = []

        # Per ogni colonna target
        for target_col in columns:
            other_cols = [c for c in columns if c != target_col]

            # Definiamo il massimo numero di colonne da combinare
            max_size = max_combination_size if max_combination_size else len(other_cols)

            found = False

            # Proviamo combinazioni di varie dimensioni
            for r in range(1, max_size + 1):
                for subset in combinations(other_cols, r):
                    temp = df[list(subset) + [target_col]].drop_duplicates()
                    grouped = temp.groupby(list(subset))[target_col].nunique()

                    if (grouped <= 1).all():
                        injective_cols_in_table.append((subset, target_col))
                        found = True
                        break  # Se troviamo una dipendenza, non cerchiamo combinazioni più grandi
                if found:
                    break

        injective_dependencies[table] = injective_cols_in_table

    conn.close()
    return injective_dependencies

def find_all_injective_dependencies(db_path, max_combination_size=None):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    injective_dependencies = {}

    # Troviamo tutte le tabelle
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]

    for table in tables:
        print(f"\nAnalizzo tabella: {table}")
        df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
        columns = df.columns.tolist()
        injective_cols_in_table = []

        # Per ogni colonna target
        for target_col in columns:
            other_cols = [c for c in columns if c != target_col]

            # Definiamo il massimo numero di colonne da combinare
            max_size = max_combination_size if max_combination_size else len(other_cols)

            # Proviamo combinazioni di varie dimensioni
            for r in range(1, max_size + 1):
                for subset in combinations(other_cols, r):
                    temp = df[list(subset) + [target_col]].drop_duplicates()
                    grouped = temp.groupby(list(subset))[target_col].nunique()

                    if (grouped <= 1).all():
                        injective_cols_in_table.append((subset, target_col))
                        # 🛑 Non facciamo break, vogliamo trovare tutte le combinazioni valide

        injective_dependencies[table] = injective_cols_in_table

    conn.close()
    return injective_dependencies
# Esempio di utilizzo:


def find_minimal_injective_dependencies(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    injective_dependencies = {}

    # Troviamo tutte le tabelle
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]

    for table in tables:
        print(f"\nAnalizzo tabella: {table}")
        df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
        columns = df.columns.tolist()
        injective_cols_in_table = []

        # Per ogni colonna target
        for target_col in columns:
            other_cols = [c for c in columns if c != target_col]

            # Partiamo da tutte le colonne tranne target
            group = other_cols.copy()

            # Verifica iniettività (controllo corretto)
            temp = df[group + [target_col]]
            duplicates = temp.duplicated(subset=group, keep=False)
            conflict = temp[duplicates].drop_duplicates(subset=group)

            if conflict.shape[0] == 0:
                minimal_group = group.copy()
                changed = True
                while changed and len(minimal_group) > 1:
                    changed = False
                    for col in minimal_group.copy():
                        test_group = [c for c in minimal_group if c != col]
                        temp = df[test_group + [target_col]]
                        duplicates = temp.duplicated(subset=test_group, keep=False)
                        conflict = temp[duplicates].drop_duplicates(subset=test_group)

                        if conflict.shape[0] == 0:
                            minimal_group = test_group
                            changed = True
                            break  # Restart after every successful removal

                injective_cols_in_table.append((tuple(minimal_group), target_col))

        injective_dependencies[table] = injective_cols_in_table

    conn.close()
    return injective_dependencies

def find_maximal_injective_dependencies(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    injective_dependencies = {}

    # Prendiamo tutte le tabelle
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]

    for table in tables:
        print(f"\nAnalizzo tabella: {table}")
        df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
        columns = df.columns.tolist()
        injective_cols_in_table = []

        # Per ogni colonna target
        for target_col in columns:
            other_cols = [c for c in columns if c != target_col]

            # 1. Parti da tutte le colonne tranne il target
            group = other_cols.copy()

            # 2. Controlla se il gruppo completo è iniettivo
            temp = df[group + [target_col]].drop_duplicates()
            grouped = temp.groupby(group)[target_col].nunique()

            if (grouped <= 1).all():
                injective_cols_in_table.append((tuple(group), target_col))
            # Se NON è iniettivo, NON esploriamo sottogruppi!
            # (scartiamo subito senza perdere tempo)

        injective_dependencies[table] = injective_cols_in_table

    conn.close()
    return injective_dependencies

result = find_minimal_injective_dependencies('data/db.sqlite')
for table, deps in result.items():
    print(f"\nTabella: {table}")
    for cols, target in deps:
        print(f"{cols} -> {target}")