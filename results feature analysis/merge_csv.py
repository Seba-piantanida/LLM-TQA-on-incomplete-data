import pandas as pd

# === Specifica i tuoi file ===
file1 = "out_test_300_sum_normal_QATCH.csv"
file2 = "out_test_simple_sum_normal_QATCH.csv"
output_file = "all_tests.csv"

# === Leggi i CSV ===
df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)

# === Unisci i DataFrame ===
df_combined = pd.concat([df1, df2], ignore_index=True)

# === Salva il risultato ===
df_combined.to_csv(output_file, index=False)

print(f"✅ File salvato in: {output_file}")