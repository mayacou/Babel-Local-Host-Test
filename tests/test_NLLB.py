import os
import csv
import pytest
from helpers.evaluation import compute_bleu, compute_comet
from models.load_NLLB import load_model, translate_text
from datasets_loader.load_europarl import load_europarl_data
from datasets_loader.load_ted import load_ted_data
from datasets_loader.load_wmt import load_wmt_data

# Define output CSV file
RESULTS_CSV = "translation_results/NLLB200_test_results.csv"

# Ensure output directory exists
os.makedirs("translation_results", exist_ok=True)

# Load model and tokenizer
model, tokenizer = load_model()

# Datasets and their loaders
DATASETS = {
    "Europarl": load_europarl_data,
    "TED": load_ted_data,
    "WMT": load_wmt_data,
}

def write_to_csv(dataset, language, bleu, comet):
    """Append a row to the results CSV file."""
    file_exists = os.path.isfile(RESULTS_CSV)
    with open(RESULTS_CSV, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Dataset", "Language", "BLEU", "COMET"])
        writer.writerow([dataset, language, bleu, comet])

@pytest.mark.parametrize("dataset_name", DATASETS.keys())
def test_translation_quality(dataset_name):
    """Test NLLB-200 translations and log results."""
    print(f"Testing NLLB-200 on {dataset_name}")
    dataset_loader = DATASETS[dataset_name]
    languages = dataset_loader("get_languages")
    
    for language in languages:
        print(f"Processing {language} in {dataset_name}")
        sources, references = dataset_loader(language)
        if not sources:
            print(f"No data for {language}, skipping.")
            continue
        
        hypotheses = [translate_text(model, tokenizer, src, "eng_Latn", f"{language}_Latn") for src in sources]
        bleu_score = compute_bleu(references, hypotheses)
        comet_score = compute_comet(references, hypotheses, sources)
        write_to_csv(dataset_name, language, round(bleu_score, 2), round(comet_score, 2))
        print(f"{dataset_name} | {language} -> BLEU: {bleu_score}, COMET: {comet_score}")

        assert bleu_score > 10, "BLEU score too low"
        assert comet_score > 0.5, "COMET score too low"
