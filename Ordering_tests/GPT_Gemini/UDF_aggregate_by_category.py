#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script per processare file CSV con join su "question" e aggregazione su model, execution_type, test_category

Versione 5.0 - Gestione execution_type vuoti/null come NULL
"""

import pandas as pd
import numpy as np
import os
import glob
import sys
import re
import ast
from difflib import SequenceMatcher

def clean_execution_type(execution_type_series):
    """
    Pulisce la colonna execution_type convertendo valori vuoti/null in NaN
    """
    print("\nðŸ§¹ Pulizia execution_type...")

    # Crea una copia della serie
    cleaned = execution_type_series.copy()

    # Converti tutto in stringa per il controllo
    cleaned_str = cleaned.astype(str).str.strip().str.lower()

    # Identifica valori da considerare come NULL
    null_values = ['', 'nan', 'none', 'null', '<na>', 'na']
    null_mask = cleaned_str.isin(null_values)

    # Sostituisci con NaN
    cleaned[null_mask] = np.nan

    # Statistiche di pulizia
    original_nulls = execution_type_series.isnull().sum()
    additional_nulls = null_mask.sum()
    final_nulls = cleaned.isnull().sum()

    print(f"  ðŸ“Š Statistiche pulizia execution_type:")
    print(f"     NULL originali: {original_nulls}")
    print(f"     Valori vuoti convertiti in NULL: {additional_nulls}")
    print(f"     Totale NULL dopo pulizia: {final_nulls}")
    print(f"     Valori validi: {len(cleaned) - final_nulls}")

    # Mostra distribuzione dei valori validi
    if final_nulls < len(cleaned):
        valid_values = cleaned.dropna().value_counts()
        print(f"  ðŸ“ˆ Distribuzione valori validi:")
        for value, count in valid_values.items():
            print(f"     {value}: {count}")

    return cleaned

def clean_question_string(question_str):
    """
    Pulisce una stringa di domanda per migliorare il matching
    """
    if pd.isna(question_str):
        return ""

    # Converti in stringa
    question_str = str(question_str)

    # Rimuovi spazi extra
    question_str = ' '.join(question_str.split())

    # Rimuovi caratteri di newline e tab
    question_str = question_str.replace('\n', ' ').replace('\t', ' ')

    # Rimuovi punteggiatura finale per matching piÃ¹ flessibile
    question_str = question_str.rstrip('.,!?;:')

    # Converti in lowercase per matching case-insensitive
    question_str = question_str.lower().strip()

    return question_str

def find_best_matching_strategy_questions(df1, col1, df2, col2):
    """
    Trova la migliore strategia di matching tra due colonne di domande
    """
    print(f"\nðŸ” Analisi strategie di matching per domande...")

    # Strategia 1: Match esatto (pulito)
    df1_clean = df1.copy()
    df2_clean = df2.copy()

    df1_clean[f'{col1}_clean'] = df1_clean[col1].apply(clean_question_string)
    df2_clean[f'{col2}_clean'] = df2_clean[col2].apply(clean_question_string)

    exact_matches = len(set(df1_clean[f'{col1}_clean']) & set(df2_clean[f'{col2}_clean']))
    print(f"âœ“ Match esatti (puliti): {exact_matches}")

    # Strategia 2: Match parziale (prime 50 parole)
    df1_clean[f'{col1}_partial'] = df1_clean[f'{col1}_clean'].apply(lambda x: ' '.join(x.split()[:50]))
    df2_clean[f'{col2}_partial'] = df2_clean[f'{col2}_clean'].apply(lambda x: ' '.join(x.split()[:50]))

    partial_matches = len(set(df1_clean[f'{col1}_partial']) & set(df2_clean[f'{col2}_partial']))
    print(f"âœ“ Match parziali (50 parole): {partial_matches}")

    # Strategia 3: Match sui primi 100 caratteri
    df1_clean[f'{col1}_short'] = df1_clean[f'{col1}_clean'].str[:100]
    df2_clean[f'{col2}_short'] = df2_clean[f'{col2}_clean'].str[:100]

    short_matches = len(set(df1_clean[f'{col1}_short']) & set(df2_clean[f'{col2}_short']))
    print(f"âœ“ Match corti (100 char): {short_matches}")

    # Decide la strategia migliore
    strategies = {
        'exact': exact_matches,
        'partial': partial_matches,
        'short': short_matches
    }

    best_strategy = max(strategies.keys(), key=lambda k: strategies[k])
    best_score = strategies[best_strategy]

    print(f"ðŸŽ¯ Strategia migliore: {best_strategy} ({best_score} matches)")

    return best_strategy, df1_clean, df2_clean

def perform_robust_join_questions(df_main, df_ref, main_col, ref_col, ref_target_col):
    """
    Esegue un join robusto sulle domande con multiple strategie
    """
    print(f"\nðŸ”— Join robusto tra {main_col} e {ref_col}...")

    # Trova la strategia migliore
    strategy, df_main_processed, df_ref_processed = find_best_matching_strategy_questions(
        df_main, main_col, df_ref, ref_col
    )

    # Applica la strategia scelta
    if strategy == 'exact':
        join_key_main = f'{main_col}_clean'
        join_key_ref = f'{ref_col}_clean'
    elif strategy == 'partial':
        join_key_main = f'{main_col}_partial'
        join_key_ref = f'{ref_col}_partial'
    else:  # short
        join_key_main = f'{main_col}_short'
        join_key_ref = f'{ref_col}_short'

    # Prepara i DataFrames per il join
    df_for_join = df_ref_processed[[ref_col, join_key_ref, ref_target_col]].copy()
    df_for_join = df_for_join.drop_duplicates(subset=[join_key_ref])

    # Esegui il join
    merged_df = df_main_processed.merge(
        df_for_join,
        left_on=join_key_main,
        right_on=join_key_ref,
        how='left',
        suffixes=('', '_ref')
    )

    # Rimuovi colonne temporanee
    cols_to_remove = [col for col in merged_df.columns 
                     if col.endswith(('_clean', '_partial', '_short', '_ref')) 
                     and col != ref_target_col]
    merged_df = merged_df.drop(columns=cols_to_remove)

    return merged_df

def process_csv_files_with_null_execution_type(results_folder, reference_file_path, output_file_path):
    """
    Processamento CSV con gestione execution_type null/vuoti
    """

    print("="*75)
    print("PROCESSAMENTO CSV - JOIN SU QUESTION + GESTIONE EXECUTION_TYPE NULL")
    print("="*75)

    # Trova tutti i file CSV
    csv_pattern = os.path.join(results_folder, "**", "*.csv")
    csv_files = glob.glob(csv_pattern, recursive=True)

    print(f"ðŸ“ Cartella: {results_folder}")
    print(f"ðŸ“„ File CSV trovati: {len(csv_files)}")

    if not csv_files:
        print("âŒ Nessun file CSV trovato!")
        return None

    # Carica e combina tutti i CSV
    all_dataframes = []
    required_columns = [
        'model', 'execution_type', 'db_path', 'table', 'sql_query', 
        'question', 'result', 'tables_used', 'valid_efficiency_score',
        'cell_precision', 'cell_recall', 'execution_accuracy', 
        'tuple_cardinality', 'tuple_constraint', 'tuple_order'
    ]

    for file_path in csv_files:
        try:
            df = pd.read_csv(file_path)
            missing_cols = [col for col in required_columns if col not in df.columns]

            if not missing_cols:
                all_dataframes.append(df)
                print(f"âœ… {os.path.basename(file_path)}: {len(df)} righe")
            else:
                print(f"âš ï¸  Saltato {os.path.basename(file_path)}: colonne mancanti {missing_cols}")

        except Exception as e:
            print(f"âŒ Errore in {os.path.basename(file_path)}: {str(e)}")

    if not all_dataframes:
        print("âŒ Nessun file valido!")
        return None

    combined_df = pd.concat(all_dataframes, ignore_index=True)
    print(f"\nðŸ“Š Dataset combinato: {len(combined_df):,} righe")

    # Verifica che esistano le colonne necessarie
    required_group_cols = ['question', 'execution_type']
    missing_group_cols = [col for col in required_group_cols if col not in combined_df.columns]

    if missing_group_cols:
        print(f"âŒ Colonne mancanti per l'aggregazione: {missing_group_cols}")
        print(f"Colonne disponibili: {list(combined_df.columns)}")
        return None

    # PULIZIA EXECUTION_TYPE - GESTIONE VALORI VUOTI/NULL
    combined_df['execution_type'] = clean_execution_type(combined_df['execution_type'])

    # Mostra distribuzione execution_type dopo pulizia
    print(f"\nðŸ“ˆ Distribuzione execution_type dopo pulizia:")
    exec_type_counts = combined_df['execution_type'].value_counts(dropna=False)
    for exec_type, count in exec_type_counts.items():
        percentage = count/len(combined_df)*100
        if pd.isna(exec_type):
            print(f"  NULL/Vuoto: {count:,} righe ({percentage:.1f}%)")
        else:
            print(f"  {exec_type}: {count:,} righe ({percentage:.1f}%)")

    # Carica file di riferimento
    try:
        reference_df = pd.read_csv(reference_file_path)
        print(f"\nðŸ“– File riferimento: {len(reference_df):,} righe")
        print(f"Colonne riferimento: {list(reference_df.columns)}")

        # Determina quale colonna usare per il join nel file di riferimento
        join_column = None
        target_column = None

        if 'question' in reference_df.columns and 'test_category' in reference_df.columns:
            join_column = 'question'
            target_column = 'test_category'
        elif 'query' in reference_df.columns and 'test_category' in reference_df.columns:
            join_column = 'query'  # Assumendo che 'query' contenga le domande
            target_column = 'test_category'
        else:
            print(f"âŒ Il file di riferimento deve avere 'question' o 'query' + 'test_category'")
            return None

        print(f"ðŸ”— Join su: question â†” {join_column}")
        print(f"ðŸ·ï¸  Target: {target_column}")

    except Exception as e:
        print(f"âŒ Errore file di riferimento: {str(e)}")
        return None

    # Debug: mostra campioni di dati
    print(f"\nðŸ” DEBUG - Campioni di dati:")
    print(f"question (prime 3):")
    for i, val in enumerate(combined_df['question'].head(3)):
        print(f"  {i+1}. {repr(str(val)[:80])}")

    print(f"\n{join_column} (prime 3):")  
    for i, val in enumerate(reference_df[join_column].head(3)):
        print(f"  {i+1}. {repr(str(val)[:80])}")

    # Esegui join robusto
    merged_df = perform_robust_join_questions(
        combined_df, reference_df, 'question', join_column, target_column
    )

    # Controlla risultato del join
    matched_rows = merged_df[target_column].notna().sum()
    print(f"\nâœ… Join completato: {matched_rows:,}/{len(merged_df):,} righe matchate ({matched_rows/len(merged_df)*100:.1f}%)")

    if matched_rows == 0:
        print("âŒ Ancora nessun match! Verifica manualmente i dati.")

        # Salva un file di debug
        debug_df = pd.DataFrame({
            'question_sample': combined_df['question'].head(10),
            f'{join_column}_sample': reference_df[join_column].head(10).tolist() + [None] * (10 - min(10, len(reference_df)))
        })
        debug_df.to_csv('debug_questions.csv', index=False)
        print("ðŸ’¾ Salvato debug_questions.csv per analisi manuale")
        return None

    # Continua con raggruppamento solo se ci sono match
    valid_df = merged_df.dropna(subset=[target_column])

    metric_columns = [
        'valid_efficiency_score', 'cell_precision', 'cell_recall', 
        'execution_accuracy', 'tuple_cardinality', 'tuple_constraint', 'tuple_order'
    ]

    # GESTIONE RAGGRUPPAMENTO CON EXECUTION_TYPE NULL
    print(f"\nðŸ“Š Analisi dati per raggruppamento...")

    # Separa righe con execution_type valido da quelle NULL
    valid_exec_type_df = valid_df.dropna(subset=['execution_type'])
    null_exec_type_df = valid_df[valid_df['execution_type'].isnull()]

    print(f"  ðŸ“Š Righe con execution_type valido: {len(valid_exec_type_df):,}")
    print(f"  âŒ Righe con execution_type NULL: {len(null_exec_type_df):,}")

    grouped_dfs = []

    # Raggruppa le righe con execution_type valido
    if len(valid_exec_type_df) > 0:
        grouping_columns = ['model', 'execution_type', target_column]
        print(f"\nðŸ”„ Raggruppamento su: {grouping_columns}")

        grouped_valid = valid_exec_type_df.groupby(grouping_columns)[metric_columns].mean().reset_index()

        # Arrotonda i risultati
        for col in metric_columns:
            grouped_valid[col] = grouped_valid[col].round(4)

        grouped_dfs.append(grouped_valid)
        print(f"âœ… Raggruppamento valido: {len(grouped_valid)} combinazioni")

    # Raggruppa le righe con execution_type NULL (solo su model e test_category)
    if len(null_exec_type_df) > 0:
        grouping_columns_null = ['model', target_column]
        print(f"\nðŸ”„ Raggruppamento righe NULL su: {grouping_columns_null}")

        grouped_null = null_exec_type_df.groupby(grouping_columns_null)[metric_columns].mean().reset_index()

        # Aggiungi execution_type come NULL esplicitamente
        grouped_null['execution_type'] = np.nan

        # Riordina colonne per consistenza
        grouped_null = grouped_null[['model', 'execution_type', target_column] + metric_columns]

        # Arrotonda i risultati
        for col in metric_columns:
            grouped_null[col] = grouped_null[col].round(4)

        grouped_dfs.append(grouped_null)
        print(f"âœ… Raggruppamento NULL: {len(grouped_null)} combinazioni")

    # Combina i risultati
    if not grouped_dfs:
        print("âŒ Nessun dato da raggruppare!")
        return None

    grouped_df = pd.concat(grouped_dfs, ignore_index=True)

    print(f"\nâœ… Raggruppamento finale: {len(grouped_df)} combinazioni totali")
    print(f"ðŸ¤– Modelli unici: {grouped_df['model'].nunique()}")
    print(f"âš™ï¸  Execution types unici (inclusi NULL): {grouped_df['execution_type'].nunique()}")
    print(f"ðŸ·ï¸  Categorie test uniche: {grouped_df[target_column].nunique()}")

    # Mostra distribuzione dettagliata
    print(f"\nðŸ“ˆ Distribuzione finale:")
    print(f"Per execution_type:")
    exec_dist = grouped_df['execution_type'].value_counts(dropna=False)
    for exec_type, count in exec_dist.items():
        if pd.isna(exec_type):
            print(f"  NULL/Vuoto: {count} combinazioni")
        else:
            print(f"  {exec_type}: {count} combinazioni")

    # Salva risultato
    try:
        grouped_df.to_csv(output_file_path, index=False)
        print(f"\nðŸ’¾ Salvato: {output_file_path}")

        print(f"\nðŸ‘€ Anteprima risultato:")
        print("-" * 120)
        print(grouped_df.head(15).to_string(index=False))

        # Salva anche un file di riepilogo
        summary_file = output_file_path.replace('.csv', '_summary.csv')

        summary_df = grouped_df.groupby(['model', 'execution_type']).agg({
            target_column: 'count',
            'execution_accuracy': ['mean', 'std'],
            'cell_precision': 'mean',
            'cell_recall': 'mean'
        }).round(4)

        summary_df.columns = ['_'.join(col).strip() if col[1] else col[0] for col in summary_df.columns.values]
        summary_df = summary_df.reset_index()
        summary_df.to_csv(summary_file, index=False)

        print(f"ðŸ“„ Salvato anche riepilogo: {summary_file}")

        return grouped_df

    except Exception as e:
        print(f"âŒ Errore salvataggio: {str(e)}")
        return None

def main():
    """
    Configurazione e esecuzione - GESTIONE EXECUTION_TYPE NULL
    """
    # CONFIGURA I PERCORSI
    results_folder = "results_UDF/UDF/classic_ds_lama_deep"
    reference_file = "tests/UDF/all_UDF_tests.csv"  
    output_file = "results_UDF/UDF/aggregated_results_lama-ds_null_execution_type.csv"

    # Verifica percorsi
    if not os.path.exists(reference_file):
        print(f"âŒ File di riferimento non trovato: {reference_file}")
        return

    # Esegui processamento
    result_df = process_csv_files_with_null_execution_type(results_folder, reference_file, output_file)

    if result_df is not None:
        print(f"\nðŸŽ‰ Completato! Risultati in: {output_file}")
        print(f"ðŸ“Š Totale combinazioni: {len(result_df)}")

        # Statistiche finali
        null_exec_count = result_df['execution_type'].isnull().sum()
        valid_exec_count = len(result_df) - null_exec_count

        print(f"\nðŸ“ˆ Statistiche finali:")
        print(f"  âœ… Combinazioni con execution_type valido: {valid_exec_count}")
        print(f"  âŒ Combinazioni con execution_type NULL: {null_exec_count}")
    else:
        print(f"\nâŒ Processamento fallito.")

if __name__ == "__main__":
    main()