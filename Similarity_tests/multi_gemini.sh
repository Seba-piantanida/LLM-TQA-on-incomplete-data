#!/bin/bash

# Array dei file CSV
csv_files=(
    
    "tests/test_movie_similarity_bilingual_cut_300.csv"
    "tests/test_movie_similarity_bilingual_cut_1000.csv"
)
# "tests/test_movie_similarity_bilingual.csv"

# Array delle cartelle di output corrispondenti
output_dirs=(
   
    "./results/no_id/cut_300"
    "./results/no_id/cut_1000"
)
#"./results/no_id/full"

# Array dei modelli
models=("gemini-2.5-flash" "gemini-2.5-pro")
# year genre directors writers main_cast duration_min AVG_score number_of_votes

# Loop su tutte le combinazioni CSV x modello
for i in "${!csv_files[@]}"; do
    csv="${csv_files[$i]}"
    output="${output_dirs[$i]}"
    
    for model in "${models[@]}"; do
        log_file="output_$(basename "$csv" .csv)_${model}.log"
        
        echo "Running: CSV=$csv -> Output=$output -> Model=$model"
        
        venv/bin/python3 gemini_run.py \
            --test_csv "$csv" \
            --modes NULL REMOVE \
            --rem_columns  IMDB_id\
            --output_dir "$output" \
            --model "$model" > "$log_file" 2>&1
        
        echo "Finished: $log_file"
        echo "---------------------------------------"
    done
done