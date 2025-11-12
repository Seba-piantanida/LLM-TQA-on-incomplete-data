#!/bin/bash

# Array dei file CSV
csv_files=(
    "tests/test_movie_similarity_bilingual_cut_300.csv"
    "tests/test_movie_similarity_bilingual_cut_300.csv"
    "tests/test_movie_similarity_bilingual_cut_300.csv"
    "tests/test_movie_similarity_bilingual_cut_300.csv"
)

# Array delle cartelle di output corrispondenti
output_dirs=(
    "results/no_all/cut_300_llama_ds"
    "results/no_title_year/cut_300_llama_ds"
    "results/normal/cut_300_llama_ds"
    "results/no_id/cut_300_llama_ds"
)

# Array delle colonne da rimuovere per ogni test
rem_columns=(
    "year genre directors writers main_cast duration_min AVG_score number_of_votes"
    "title year"
    ""
    "IMDB_id"
)

# Array delle modalità per ogni test
test_modes=(
    "NULL REMOVE"
    "NULL REMOVE"
    "NORMAL"
    "NULL REMOVE"
    
)

# Array dei modelli Together AI
models=(
    "deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free"
)

# Loop su tutte le combinazioni CSV x modello
for i in "${!csv_files[@]}"; do
    csv="${csv_files[$i]}"
    output="${output_dirs[$i]}"
    rem_col="${rem_columns[$i]}"
    modes="${test_modes[$i]}"
    
    for model in "${models[@]}"; do
        # Crea nome file log più pulito
        test_name=$(basename "$csv" .csv)
        model_safe=$(echo "$model" | sed 's/[\/\-]/_/g')
        test_type=$(basename "$output")
        log_file="logs/output_${test_name}_${test_type}_${model_safe}.log"
        
        # Crea directory logs se non esiste
        mkdir -p logs
        
        echo "Running Test #$((i+1))/4"
        echo "  CSV: $csv"
        echo "  Output: $output"
        echo "  Model: $model"
        echo "  Modes: $modes"
        if [ -n "$rem_col" ]; then
            echo "  Removing columns: $rem_col"
        else
            echo "  Mode: NORMAL (no columns removed)"
        fi
        
        # Costruisci il comando base
        cmd="venv/bin/python3 Together_AI_run.py --test_csv \"$csv\" --modes $modes --output_dir \"$output\" --model \"$model\""
        
        # Aggiungi --rem_columns solo se ci sono colonne da rimuovere
        if [ -n "$rem_col" ]; then
            cmd="$cmd --rem_columns $rem_col"
        fi
        
        # Esegui il comando
        eval "$cmd" > "$log_file" 2>&1
        
        echo "  ✓ Finished: $log_file"
        echo "---------------------------------------"
        echo ""
    done
done

echo "================================================"
echo "All tests completed!"
echo "================================================"
echo "Total tests executed: $((${#csv_files[@]} * ${#models[@]}))"
echo "Logs saved in: logs/"
echo "================================================"