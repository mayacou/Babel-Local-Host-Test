# pytest -v -s tests/test_NLLB.py

import os
import csv
import pytest
import torch
from helpers.evaluation import compute_bleu, compute_comet
from models.load_NLLB import load_model, translate_text
from datasets_loader.load_europarl import load_europarl_data
from datasets_loader.load_tedTalk import load_tedTalk_data
from datasets_loader.load_wmt import load_wmt_data

# Define output CSV files
RESULTS_CSV = "data/NLLB200_test_results.csv"
TRANSLATIONS_CSV = "translation_results/NLLB200_translations.csv"

# Ensure output directory exists
os.makedirs("translation_results", exist_ok=True)

LANGUAGE_CODE_MAP = {
    "bg": "bul_Cyrl", "cs": "ces_Latn", "da": "dan_Latn", "nl": "nld_Latn", "et": "est_Latn", 
    "fi": "fin_Latn", "fr": "fra_Latn", "de": "deu_Latn", "el": "ell_Latn", "hu": "hun_Latn", 
    "it": "ita_Latn", "lv": "lav_Latn", "lt": "lit_Latn", "pl": "pol_Latn", "pt": "por_Latn", 
    "ro": "ron_Latn", "sk": "slk_Latn", "sl": "slv_Latn", "es": "spa_Latn", "sv": "swe_Latn", 
    "tr": "tur_Latn", "hr": "hrv_Latn", "is": "isl_Latn", "mk": "mkd_Cyrl", "sq": "sqi_Latn", 
    "no": "nob_Latn"
}

# Load model and tokenizer
model, tokenizer, device = load_model()

# Datasets and their loaders
DATASETS = {
    "Europarl": load_europarl_data,
    "TED": load_tedTalk_data,
    "WMT": load_wmt_data,
}

@pytest.mark.parametrize("dataset_name", DATASETS.keys())
def test_translation_quality(dataset_name):
    """Test NLLB-200 translations and log results."""
    print(f"Testing NLLB-200 on {dataset_name}")
    dataset_loader = DATASETS[dataset_name]

    # Fetch languages from dataset
    languages = dataset_loader("get_languages")

    # Open CSV files **once** for the entire test
    with open(RESULTS_CSV, mode="a", newline="") as results_file, open(TRANSLATIONS_CSV, mode="a", newline="") as trans_file:
        results_writer = csv.writer(results_file)
        trans_writer = csv.writer(trans_file)

        # Write headers if file is new
        if os.stat(RESULTS_CSV).st_size == 0:
            results_writer.writerow(["Dataset", "Language", "BLEU", "COMET"])

        if os.stat(TRANSLATIONS_CSV).st_size == 0:
            trans_writer.writerow(["Dataset", "Language", "Source Sentence", "Translation", "Reference Sentence"])

        for language in languages:
            if language not in LANGUAGE_CODE_MAP:
                print(f"⚠️ Skipping {language}: No mapping found for NLLB-200.")
                continue

            nllb_language_code = LANGUAGE_CODE_MAP[language]  
            print(f"Processing {language} ({nllb_language_code}) in {dataset_name}")
            
            sources, references = dataset_loader(language)
            if not sources:
                print(f"⚠️ No data for {language}, skipping.")
                continue

            # Translate using mapped language code
            hypotheses = [translate_text(model, tokenizer, src, "eng_Latn", nllb_language_code,device) for src in sources]

            # Evaluate translation quality
            bleu_score = compute_bleu(references, hypotheses)
            comet_score = compute_comet(references, hypotheses, sources)

            # Save results to CSV
            results_writer.writerow([dataset_name, language, round(bleu_score, 2), round(comet_score, 2)])

            for src, hyp, ref in zip(sources, hypotheses, references):
                trans_writer.writerow([dataset_name, language, src, hyp, ref])

            print(f"✅ {dataset_name} | {language} ({nllb_language_code}) -> BLEU: {bleu_score}, COMET: {comet_score}")

        # Ensure data is flushed to file
        results_file.flush()
        trans_file.flush()
