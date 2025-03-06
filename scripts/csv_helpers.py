import os
import csv

def write_to_csv(path , model, language, bleu, comet):
    """Append a row to the CSV file."""
    file_exists = os.path.isfile(path)

    with open(path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Model", "Language", "BLEU", "COMET"])  
        writer.writerow([model, language, bleu, comet])
