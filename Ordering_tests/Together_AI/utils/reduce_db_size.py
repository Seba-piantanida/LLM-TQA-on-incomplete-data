import sqlite3

def reduce_tables(original_db, new_db, X):
    """
    Copies all tables from the original database to a new database,
    reducing the number of rows in each table to X / number of columns.
    """
    try:
        
        conn_orig = sqlite3.connect(original_db)
        cursor_orig = conn_orig.cursor()
        
        conn_new = sqlite3.connect(new_db)
        cursor_new = conn_new.cursor()

        
        cursor_orig.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [table[0] for table in cursor_orig.fetchall() if table[0] != 'sqlite_sequence']

        for table_name in tables:
            
            cursor_orig.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name=?",
                (table_name,)
            )
            create_table = cursor_orig.fetchone()
            if not create_table or not create_table[0]:
                continue  
            
            
            cursor_new.execute(create_table[0])

            
            cursor_orig.execute(f'PRAGMA table_info("{table_name}")')
            num_columns = len(cursor_orig.fetchall())
            if num_columns == 0:
                continue  

           
            num_rows = X // num_columns
            if num_rows <= 0:
                continue  

           
            cursor_orig.execute(
                f'SELECT * FROM "{table_name}" LIMIT ?',
                (num_rows,)
            )
            rows = cursor_orig.fetchall()

            
            if rows:
                placeholders = ', '.join(['?'] * num_columns)
                insert_query = (
                    f'INSERT INTO "{table_name}" VALUES ({placeholders})'
                )
                cursor_new.executemany(insert_query, rows)

        conn_new.commit()
        print("Operazione completata con successo!")

    except sqlite3.Error as e:
        print(f"Errore SQLite: {e}")
    except Exception as e:
        print(f"Errore generico: {e}")
    finally:
        if 'conn_orig' in locals():
            conn_orig.close()
        if 'conn_new' in locals():
            conn_new.close()


original_db = "data/music.sqlite"

cut_size = 300  # n_of_col x n_of_row
new_db = f"data/db_music_cut_{cut_size}.sqlite"

reduce_tables(original_db, new_db, cut_size)