import os
import csv
from config.languages import LANGUAGES

def write_results_to_csv(path, model, dataset, language, bleu, comet):
    """Append a row to the results CSV file."""
    file_exists = os.path.isfile(path)

    # Get the full language name from the LANGUAGES dictionary
    language_name = LANGUAGES.get(language, language)  # If language not found, use the code as is

    with open(path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Model", "Dataset", "Language", "BLEU", "COMET"])  # Write header only if file doesn't exist
        writer.writerow([model, dataset, language_name, bleu, comet])  # Write the results row
def write_all_translations_to_csv(sources, hypotheses, references, csv_file, src_lang, tgt_lang):
    try:
        with open(csv_file, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            
            # Write the header if it's the first write (to avoid writing headers repeatedly)
            if file.tell() == 0:
                writer.writerow(["Source", "Hypothesis", "Reference", "Source Language", "Target Language"])

            # Write translations to CSV
            for source, hyp, ref in zip(sources, hypotheses, references):
                writer.writerow([source, hyp, ref, src_lang, tgt_lang])

        print(f"✅ All translations saved to {csv_file}")
    except Exception as e:
        print(f"❌ Error writing translations to CSV: {e}")