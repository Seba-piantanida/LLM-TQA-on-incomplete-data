import sqlite3
import pandas as pd
import re
import shutil
import os

def check_database_tracks(db_path):
    """
    Controlla il contenuto della tabella tracks per identificare valori problematici
    """
    try:
        conn = sqlite3.connect(db_path)
        
        # Ottieni informazioni sulla struttura della tabella
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(tracks)")
        columns_info = cursor.fetchall()
        
        print("Struttura della tabella 'tracks':")
        for col in columns_info:
            print(f"  {col[1]} ({col[2]}) - NOT NULL: {col[3]}")
        print()
        
        # Carica alcuni record per vedere i dati
        df = pd.read_sql_query("SELECT * FROM tracks LIMIT 10", conn)
        print("Primi 10 record della tabella 'tracks':")
        print(df)
        print()
        
        # Cerca colonne che potrebbero contenere valori con virgole
        for col in df.columns:
            if df[col].dtype == 'object':  # colonne di testo
                # Cerca valori che contengono virgole e sembrano numeri
                problematic_values = df[df[col].astype(str).str.contains(r'^\d+,\d+$', na=False)][col]
                if not problematic_values.empty:
                    print(f"Valori problematici trovati nella colonna '{col}':")
                    print(problematic_values.tolist())
                    print()
        
        # Cerca specificamente il valore '1,103'
        for col in df.columns:
            try:
                has_problematic = df[df[col].astype(str) == '1,103']
                if not has_problematic.empty:
                    print(f"Valore '1,103' trovato nella colonna '{col}'")
                    print(has_problematic)
                    print()
            except:
                continue
                
        conn.close()
        
    except Exception as e:
        print(f"Errore nel controllare il database: {e}")

def check_all_tables_for_comma_values(db_path):
    """
    Controlla tutte le tabelle del database per valori con virgole
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Ottieni tutte le tabelle
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        
        print(f"Controllando {len(tables)} tabelle nel database...")
        print()
        
        for table_name in tables:
            table_name = table_name[0]
            print(f"=== Tabella: {table_name} ===")
            
            try:
                # Ottieni informazioni sulle colonne
                cursor.execute(f"PRAGMA table_info(`{table_name}`)")
                columns = cursor.fetchall()
                
                # Conta i record totali
                cursor.execute(f"SELECT COUNT(*) FROM `{table_name}`")
                total_records = cursor.fetchone()[0]
                print(f"Record totali: {total_records}")
                
                found_issues = False
                for col in columns:
                    col_name = col[1]
                    col_type = col[2].upper()
                    
                    # Cerca valori con virgole in qualsiasi colonna
                    cursor.execute(f"SELECT COUNT(*) FROM `{table_name}` WHERE `{col_name}` LIKE '%,%'")
                    comma_count = cursor.fetchone()[0]
                    
                    if comma_count > 0:
                        print(f"  Colonna '{col_name}' ({col_type}): {comma_count} valori con virgole")
                        
                        # Mostra alcuni esempi
                        cursor.execute(f"SELECT DISTINCT `{col_name}` FROM `{table_name}` WHERE `{col_name}` LIKE '%,%' LIMIT 5")
                        examples = cursor.fetchall()
                        print(f"    Esempi: {[ex[0] for ex in examples]}")
                        found_issues = True
                
                if not found_issues:
                    print("  Nessun valore con virgole trovato")
                print()
                
            except Exception as e:
                print(f"  Errore nel controllare la tabella {table_name}: {e}")
                print()
        
        conn.close()
        
    except Exception as e:
        print(f"Errore nel controllare il database: {e}")

def clean_database_numeric_values(db_path, backup_path=None):
    """
    Pulisce i valori numerici nel database rimuovendo le virgole dai separatori delle migliaia
    """
    if backup_path:
        # Crea un backup
        try:
            shutil.copy2(db_path, backup_path)
            print(f"Backup creato: {backup_path}")
        except Exception as e:
            print(f"Errore nella creazione del backup: {e}")
            return False
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    changes_made = 0
    
    try:
        # Ottieni tutte le tabelle
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        
        for table_name in tables:
            table_name = table_name[0]
            print(f"Processando tabella: {table_name}")
            
            # Ottieni informazioni sulle colonne
            cursor.execute(f"PRAGMA table_info(`{table_name}`)")
            columns = cursor.fetchall()
            
            for col in columns:
                col_name = col[1]
                col_type = col[2].upper()
                
                try:
                    # Trova tutti i record con virgole in questa colonna
                    cursor.execute(f"SELECT rowid, `{col_name}` FROM `{table_name}` WHERE `{col_name}` LIKE '%,%'")
                    problematic_rows = cursor.fetchall()
                    
                    if problematic_rows:
                        print(f"  Trovati {len(problematic_rows)} valori problematici nella colonna {col_name} ({col_type})")
                        
                        for rowid, value in problematic_rows:
                            if isinstance(value, str):
                                # Rimuovi le virgole dai separatori delle migliaia
                                cleaned_value = re.sub(r'(\d),(\d)', r'\1\2', str(value))
                                
                                # Se la colonna dovrebbe essere numerica, prova a convertire
                                if any(numeric_type in col_type for numeric_type in ['INTEGER', 'REAL', 'NUMERIC']):
                                    try:
                                        if '.' in cleaned_value:
                                            test_value = float(cleaned_value)
                                        else:
                                            test_value = int(cleaned_value)
                                        
                                        # Aggiorna il record
                                        cursor.execute(f"UPDATE `{table_name}` SET `{col_name}` = ? WHERE rowid = ?", 
                                                     (cleaned_value, rowid))
                                        print(f"    Corretto: '{value}' -> '{cleaned_value}'")
                                        changes_made += 1
                                        
                                    except ValueError:
                                        print(f"    Attenzione: Non posso convertire '{value}' in numero nella tabella {table_name}, colonna {col_name}")
                                else:
                                    # Per colonne di testo, aggiorna comunque se sembra essere un numero
                                    if re.match(r'^\d{1,3}(,\d{3})*(\.\d+)?$', value):
                                        cursor.execute(f"UPDATE `{table_name}` SET `{col_name}` = ? WHERE rowid = ?", 
                                                     (cleaned_value, rowid))
                                        print(f"    Corretto (testo): '{value}' -> '{cleaned_value}'")
                                        changes_made += 1
                                        
                except Exception as e:
                    print(f"  Errore processando colonna {col_name}: {e}")
        
        if changes_made > 0:
            conn.commit()
            print(f"Pulizia completata! {changes_made} valori corretti.")
        else:
            print("Nessuna modifica necessaria.")
        
        return True
        
    except Exception as e:
        print(f"Errore durante la pulizia: {e}")
        conn.rollback()
        return False
    
    finally:
        conn.close()

def verify_cleaning(db_path):
    """
    Verifica che la pulizia sia stata effettuata correttamente
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        
        total_issues = 0
        for table_name in tables:
            table_name = table_name[0]
            
            cursor.execute(f"PRAGMA table_info(`{table_name}`)")
            columns = cursor.fetchall()
            
            for col in columns:
                col_name = col[1]
                cursor.execute(f"SELECT COUNT(*) FROM `{table_name}` WHERE `{col_name}` LIKE '%,%'")
                count = cursor.fetchone()[0]
                if count > 0:
                    print(f"Attenzione: {count} valori con virgole ancora presenti in {table_name}.{col_name}")
                    # Mostra alcuni esempi
                    cursor.execute(f"SELECT DISTINCT `{col_name}` FROM `{table_name}` WHERE `{col_name}` LIKE '%,%' LIMIT 3")
                    examples = cursor.fetchall()
                    print(f"  Esempi: {[ex[0] for ex in examples]}")
                    total_issues += count
        
        if total_issues == 0:
            print("✅ Verifica completata: nessun problema rilevato!")
        else:
            print(f"⚠️  Verifica completata: {total_issues} problemi ancora presenti")
            
    finally:
        conn.close()

def main():
    """
    Funzione principale che esegue tutti i controlli e le pulizie
    """
    # Configura i percorsi
    db_path = 'data/music.sqlite'
    backup_path = 'data/music_backup.sqlite'
    
    # Verifica che il database esista
    if not os.path.exists(db_path):
        print(f"❌ Errore: Il database {db_path} non esiste!")
        print("Assicurati che il percorso sia corretto.")
        return
    
    print("🔍 CONTROLLO INIZIALE - Tabella tracks")
    print("=" * 50)
    check_database_tracks(db_path)
    
    print("\n🔍 CONTROLLO COMPLETO - Tutte le tabelle")
    print("=" * 50)
    check_all_tables_for_comma_values(db_path)
    
    # Chiedi conferma prima di procedere con la pulizia
    response = input("\n❓ Vuoi procedere con la pulizia automatica? (s/n): ").lower().strip()
    if response not in ['s', 'si', 'y', 'yes']:
        print("Operazione annullata dall'utente.")
        return
    
    print("\n🧹 PULIZIA DATI")
    print("=" * 50)
    success = clean_database_numeric_values(db_path, backup_path)
    
    if success:
        print("\n✅ VERIFICA FINALE")
        print("=" * 50)
        verify_cleaning(db_path)
    else:
        print("❌ La pulizia ha fallito.")

if __name__ == "__main__":
    main()