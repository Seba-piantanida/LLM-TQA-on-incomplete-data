import os
import pandas as pd
import numpy as np
from typing import List

def calculate_ndcg(relevant_scores: List[float], k: int = 10) -> float:
    """Calcola NDCG@k."""
    if not relevant_scores or k <= 0:
        return 0.0

    scores = relevant_scores[:k]
    if not scores:
        return 0.0

    dcg = scores[0] + sum(score / np.log2(i + 1) for i, score in enumerate(scores[1:], 2))
    ideal_scores = [k - i for i in range(k)]
    idcg = ideal_scores[0] + sum(score / np.log2(i + 1) for i, score in enumerate(ideal_scores[1:], 2))

    return dcg / idcg if idcg > 0 else 0.0


def process_and_update_csv_files(root_dir: str, dry_run: bool = False):
    """
    Processa tutti i CSV nella cartella e sottocartelle, ricalcola NDCG@10
    e aggiorna i file originali.
    
    Args:
        root_dir: Cartella radice contenente i CSV
        dry_run: Se True, mostra solo cosa verrebbe fatto senza modificare i file
    """
    total_files = 0
    updated_files = 0
    skipped_files = 0
    
    for dirpath, _, filenames in os.walk(root_dir):
        for file in filenames:
            if not file.endswith(".csv"):
                continue

            file_path = os.path.join(dirpath, file)
            total_files += 1
            print(f"\n{'[DRY RUN] ' if dry_run else ''}📄 Elaborazione: {file_path}")

            try:
                # Leggi il CSV mantenendo "NULL" come stringa
                df = pd.read_csv(file_path, na_values=[], keep_default_na=False)
                
                # Verifica che ground_truth sia presente
                if "ground_truth" not in df.columns:
                    print(f"⚠️ Colonna 'ground_truth' mancante. Salto questo file.")
                    skipped_files += 1
                    continue
                
                # Determina quale colonna usare per predicted_ids
                if "predicted_ids_str" in df.columns:
                    pred_col = "predicted_ids_str"
                elif "predicted_ids" in df.columns:
                    pred_col = "predicted_ids"
                elif "result" in df.columns:
                    pred_col = "result"
                    print(f"  ℹ️ Usando 'result' al posto di 'predicted_ids'")
                else:
                    print(f"⚠️ Nessuna colonna valida trovata (predicted_ids, predicted_ids_str, result). Salto questo file.")
                    skipped_files += 1
                    continue
                
                # Aggiungi colonna ndcg_10 se non esiste
                if "ndcg_10" not in df.columns:
                    df["ndcg_10"] = 0.0
                    print("  ➕ Colonna 'ndcg_10' aggiunta")
                
                # Ricalcola NDCG@10 per ogni riga
                updated_count = 0
                for idx, row in df.iterrows():
                    try:
                        # Estrai ground truth e predicted IDs
                        ground_truth_str = str(row["ground_truth"])
                        predicted_str = str(row[pred_col])
                        
                        # Gestisci valori vuoti o NaN
                        if ground_truth_str in ["", "nan", "None"] or predicted_str in ["", "nan", "None"]:
                            df.at[idx, "ndcg_10"] = 0.0
                            continue
                        
                        ground_truth_ids = ground_truth_str.split("|")
                        predicted_ids = predicted_str.split("|")
                        
                        # Calcola relevance scores
                        k = len(ground_truth_ids)
                        relevance_scores = [
                            k - i if pid in ground_truth_ids else 0 
                            for i, pid in enumerate(predicted_ids)
                        ]
                        
                        # Calcola NDCG@10
                        ndcg_10 = calculate_ndcg(relevance_scores, 10)
                        
                        # Aggiorna il valore
                        old_value = df.at[idx, "ndcg_10"]
                        df.at[idx, "ndcg_10"] = round(ndcg_10, 6)
                        
                        if abs(old_value - ndcg_10) > 0.0001:
                            updated_count += 1
                        
                    except Exception as e:
                        print(f"  ⚠️ Errore alla riga {idx}: {e}")
                        df.at[idx, "ndcg_10"] = 0.0
                        continue
                
                # Salva il file aggiornato (solo se non è dry run)
                if not dry_run:
                    df.to_csv(file_path, index=False, encoding="utf-8")
                    print(f"  ✅ File aggiornato: {updated_count} righe modificate")
                    updated_files += 1
                else:
                    print(f"  ℹ️ [DRY RUN] File verrebbe aggiornato: {updated_count} righe da modificare")
                
            except Exception as e:
                print(f"  ❌ Errore nel processare {file_path}: {e}")
                skipped_files += 1
                continue
    
    # Riepilogo finale
    print(f"\n{'='*80}")
    print(f"{'[DRY RUN] ' if dry_run else ''}RIEPILOGO:")
    print(f"  File totali processati: {total_files}")
    if not dry_run:
        print(f"  File aggiornati: {updated_files}")
    else:
        print(f"  File da aggiornare: {updated_files}")
    print(f"  File saltati: {skipped_files}")
    print(f"{'='*80}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Ricalcola NDCG@10 per tutti i CSV in una cartella e aggiorna i file originali.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Esempi di utilizzo:
    # Modalità dry-run (mostra solo cosa verrebbe fatto)
    python script.py /path/to/csv/folder --dry-run
    
    # Aggiorna effettivamente i file
    python script.py /path/to/csv/folder
        """
    )
    parser.add_argument(
        "root_dir", 
        help="Cartella di input contenente i file CSV (incluse sottocartelle)"
    )
    parser.add_argument(
        "--dry-run", 
        action="store_true",
        help="Mostra cosa verrebbe fatto senza modificare i file"
    )

    args = parser.parse_args()
    
    if not os.path.exists(args.root_dir):
        print(f"❌ Errore: La cartella '{args.root_dir}' non esiste.")
        exit(1)
    
    if args.dry_run:
        print("⚠️ MODALITÀ DRY RUN - I file non verranno modificati\n")
    
    process_and_update_csv_files(args.root_dir, args.dry_run)