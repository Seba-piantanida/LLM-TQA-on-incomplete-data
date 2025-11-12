import csv

def duplica_righe_csv(input_file, output_file, n):
    with open(input_file, mode='r', newline='', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        header = next(reader)  # Legge l'intestazione

        righe = list(reader)

    with open(output_file, mode='w', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(header)  # Scrive l'intestazione

        for riga in righe:
            for _ in range(n):
                writer.writerow(riga)

if __name__ == '__main__':
    input_file = input("Inserisci il nome del file CSV di input (es. dati.csv): ")
    output_file = 'variance_test/tests.csv'
    
    
    n = int(input("Quante volte vuoi duplicare ciascuna riga? "))
            

    duplica_righe_csv(input_file, output_file, n)
    print(f"File creato: {output_file} con ogni riga duplicata {n} volte.")