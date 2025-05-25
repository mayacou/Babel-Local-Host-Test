import os
import csv
import pytest
import torch
from helpers.evaluation import compute_bleu, compute_comet
from models.load_M2M import load_model, translate_text
from datasets_loader.load_europarl import load_europarl_data
from datasets_loader.load_tedTalk import load_tedTalk_data
from datasets_loader.load_wmt import load_wmt_data

# ✅ Define output CSV files
RESULTS_CSV = "data/scores/M2M100_test_results.csv"
TRANSLATIONS_CSV = "data/translations/M2M100_translations.csv"

# ✅ Ensure output directories exist
os.makedirs("translation_results", exist_ok=True)
os.makedirs("data", exist_ok=True)

# ✅ Language Code Mapping
LANGUAGE_CODE_MAP = {
    "bg": "bg", "cs": "cs", "da": "da", "nl": "nl", "et": "et", 
    "fi": "fi", "fr": "fr", "de": "de", "el": "el", "hu": "hu", 
    "it": "it", "lv": "lv", "lt": "lt", "pl": "pl", "pt": "pt", 
    "ro": "ro", "sk": "sk", "sl": "sl", "es": "es", "sv": "sv", 
    "tr": "tr", "hr": "hr", "is": "is", "mk": "mk", "sq": "sq", 
    "no": "no"
}  

# ✅ Load model and tokenizer once
model, tokenizer, device = load_model()

# ✅ Datasets and their loaders
DATASETS = {
    "Europarl": load_europarl_data,
    "TED": load_tedTalk_data,
    "WMT": load_wmt_data,
}

# ✅ Open CSV files and keep them open
with open(RESULTS_CSV, mode="w", newline="") as results_file, open(TRANSLATIONS_CSV, mode="w", newline="") as translations_file:
    results_writer = csv.writer(results_file)
    translations_writer = csv.writer(translations_file)

    # ✅ Write headers only once
    results_writer.writerow(["Dataset", "Language", "BLEU", "COMET"])
    translations_writer.writerow(["Dataset", "Language", "Source Sentence", "Translation", "Reference Sentence"])

    # ✅ Iterate through datasets
    for dataset_name, dataset_loader in DATASETS.items():
        print(f"🔹 Testing M2M-100 on {dataset_name} dataset...")
        
        # ✅ Get available languages for this dataset
        languages = dataset_loader("get_languages")

        for language in languages:
            if language not in LANGUAGE_CODE_MAP:
                print(f"⚠️ Skipping {language}: No mapping found for M2M-100.")
                continue

            m2m_language_code = LANGUAGE_CODE_MAP[language]  
            print(f"Processing {language} ({m2m_language_code}) in {dataset_name}")
            
            sources, references = dataset_loader(language)
            if not sources:
                print(f"⚠️ No data for {language}, skipping.")
                continue

            # ✅ Translate using mapped language code
            hypotheses = [translate_text(model, tokenizer, src, "en", m2m_language_code, device) for src in sources]
            
            # ✅ Evaluate translation quality
            bleu_score = compute_bleu(references, hypotheses)
            comet_score = compute_comet(references, hypotheses, sources)
            
            # ✅ Write to CSV results
            results_writer.writerow([dataset_name, language, round(bleu_score, 2), round(comet_score, 2)])
            results_file.flush()  # ✅ Ensure data is written immediately

            # ✅ Write translations to CSV
            for src, hyp, ref in zip(sources, hypotheses, references):
                translations_writer.writerow([dataset_name, language, src, hyp, ref])
            translations_file.flush()  # ✅ Ensure data is written immediately
            
            print(f"✅ {dataset_name} | {language} ({m2m_language_code}) -> BLEU: {bleu_score}, COMET: {comet_score}")
