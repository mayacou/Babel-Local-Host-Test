import os
import csv
from config.languages import LANGUAGES

def write_to_csv(path, model, dataset, language, bleu, comet):
    """Append a row to the CSV file."""
    file_exists = os.path.isfile(path)

    # Get the full language name from the LANGUAGES dictionary
    language_name = LANGUAGES.get(language, language)  # If language not found, use the code as is

    with open(path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Model", "Dataset", "Language", "BLEU", "COMET"])  # Fixed missing comma
        writer.writerow([model, dataset, language_name, bleu, comet])
