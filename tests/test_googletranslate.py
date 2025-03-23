from google.cloud import translate_v2 as translate
import csv
import os
from dotenv import load_dotenv
from helpers.evaluation import compute_bleu, compute_comet
from datasets_loader.load_wmt import load_wmt_data
from datasets_loader.load_tedTalk import load_tedTalk_data
from datasets_loader.load_europarl import load_europarl_data

load_dotenv()
client = translate.Client()

# Ensure directories exist
os.makedirs("data", exist_ok=True)
os.makedirs("translation_results", exist_ok=True)

# Datasets to test
DATASETS = {
   "WMT": load_wmt_data,
   "TED": load_tedTalk_data,
   "Europarl": load_europarl_data
}

RESULTS_CSV = "data/GoogleCloud_test_results.csv"
TRANSLATIONS_CSV = "translation_results/GoogleCloud_translations.csv"

def write_to_csv(dataset, language, bleu, comet):
    """Append a row to the results CSV file."""
    with open(RESULTS_CSV, mode="a", newline="") as file:
        writer = csv.writer(file)
        if file.tell() == 0:  # Write header only if the file is empty
            writer.writerow(["Dataset", "Language", "BLEU", "COMET"])
        writer.writerow([dataset, language, bleu, comet])

def write_translations_to_csv(dataset, language, sources, hypotheses, references):
    """Append source sentences, translations, and references to a separate CSV file."""
    with open(TRANSLATIONS_CSV, mode="a", newline="") as file:
        writer = csv.writer(file)
        if file.tell() == 0:  # Write header only if the file is empty
            writer.writerow(["Dataset", "Language", "Source Sentence", "Translation", "Reference Sentence"])
        for source, hypothesis, reference in zip(sources, hypotheses, references):
            writer.writerow([dataset, language, source, hypothesis, reference])

def translate(source_sentences, target_language, reference_sentences):
    """Translate text using Google Cloud Translation API."""
    try:
        translations = [client.translate(sentence, target_language=target_language)["translatedText"] for sentence in source_sentences]
        bleu = compute_bleu(reference_sentences, translations)
        comet = compute_comet(reference_sentences, translations, source_sentences)
        return translations, bleu, comet
    except Exception as e:
        print(f"⚠️ Translation error for {target_language}: {e}")
        return ["ERROR"] * len(source_sentences), 0, 0  # Return error placeholders

for dataset_name, dataset_loader in DATASETS.items():
    print(f"🔹 Testing Google Cloud Translate on {dataset_name}")

    # Get available language pairs from dataset loader
    try:
        language_pairs = dataset_loader("get_languages")
    except Exception as e:
        print(f"⚠️ Skipping {dataset_name}: Could not fetch language pairs ({e})")
        continue

    for language in language_pairs:
        print(f"🔄 Testing Google Cloud Translate on {dataset_name} ({language})")

        try:
            sources, references = dataset_loader(language)
            if not sources:
                print(f"⚠️ Skipping {language} for {dataset_name}: No data available.")
                write_to_csv(dataset_name, language, "NA", "NA")
                continue
        except Exception as e:
            print(f"⚠️ Skipping {dataset_name} ({language}): Dataset loading error ({e})")
            write_to_csv(dataset_name, language, "NA", "NA")
            continue

        # Translate sentences with progress logging
        translations, bleu, comet = translate(sources, language, references)

        print(f"📊 BLEU: {round(bleu, 2)}, COMET: {round(comet, 2)}")

        # Save results
        write_to_csv(dataset_name, language, round(bleu, 2), round(comet, 2))
        write_translations_to_csv(dataset_name, language, sources, translations, references)

        print(f"✅ Saved results for Google Cloud Translate on {dataset_name} ({language})")
