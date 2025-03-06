import os
import csv
import pytest
from helpers.evaluation import compute_bleu, compute_comet
from models.load_M2M import load_model, translate_text
from datasets_loader.load_europarl import load_europarl_data
from datasets_loader.load_tedTalk import load_tedTalk_data
from datasets_loader.load_wmt import load_wmt_data

# Define output CSV file
RESULTS_CSV = "translation_results/M2M100_test_results.csv"  # Changed from NLLB200

# Ensure output directory exists
os.makedirs("translation_results", exist_ok=True)

LANGUAGE_CODE_MAP = {
    "bg": "bg", "cs": "cs", "da": "da", "nl": "nl", "et": "et", 
    "fi": "fi", "fr": "fr", "de": "de", "el": "el", "hu": "hu", 
    "it": "it", "lv": "lv", "lt": "lt", "pl": "pl", "pt": "pt", 
    "ro": "ro", "sk": "sk", "sl": "sl", "es": "es", "sv": "sv", 
    "tr": "tr", "hr": "hr", "is": "is", "mk": "mk", "sq": "sq", 
    "no": "no"
}  # Updated codes for M2M-100 format

# Load model and tokenizer
model, tokenizer = load_model()

# Datasets and their loaders
DATASETS = {
    "Europarl": load_europarl_data,
    "TED": load_tedTalk_data,
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
    """Test M2M-100 translations and log results."""
    print(f"Testing M2M-100 on {dataset_name}")
    dataset_loader = DATASETS[dataset_name]
    
    # Fetch languages from dataset
    languages = dataset_loader("get_languages")

    for language in languages:
        if language not in LANGUAGE_CODE_MAP:
            print(f"⚠️ Skipping {language}: No mapping found for M2M-100.")
            continue

        m2m_language_code = LANGUAGE_CODE_MAP[language]  # Convert dataset code to M2M-100 format

        print(f"Processing {language} ({m2m_language_code}) in {dataset_name}")
        
        sources, references = dataset_loader(language)
        if not sources:
            print(f"⚠️ No data for {language}, skipping.")
            continue

        # Translate using mapped language code
        hypotheses = [translate_text(model, tokenizer, src, "en", m2m_language_code) for src in sources]
        
        # Evaluate translation quality
        bleu_score = compute_bleu(references, hypotheses)
        comet_score = compute_comet(references, hypotheses, sources)
        
        # Save results
        write_to_csv(dataset_name, language, round(bleu_score, 2), round(comet_score, 2))
        print(f"✅ {dataset_name} | {language} ({m2m_language_code}) -> BLEU: {bleu_score}, COMET: {comet_score}")
