import csv
from models.load_gemini import load_gemini, translate_text
from helpers.evaluation import compute_bleu, compute_comet
from datasets_loader.load_wmt import load_wmt_data
from datasets_loader.load_tedTalk import load_tedTalk_data
from datasets_loader.load_opus import load_opus_data
from datasets_loader.load_europarl import load_europarl_data

# Datasets to test
DATASETS = {
    "WMT": load_wmt_data,
    "TED": load_tedTalk_data,
    # "OPUS": load_opus_data,
    "Europarl": load_europarl_data
}

# Load Gemini model
print("🔄 Loading Gemini model...")
model = load_gemini()
print("✅ Gemini model loaded successfully!")

# Define CSV paths
csv_filename = "data/gemini_results.csv"
translations_csv_filename = "translation_results/gemini_translations.csv"

# Ensure directories exist
import os
os.makedirs("data", exist_ok=True)
os.makedirs("translation_results", exist_ok=True)

def write_to_csv(dataset, language, bleu, comet):
    """Append a row to the results CSV file."""
    with open(csv_filename, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if file.tell() == 0:  # Write header only if the file is empty
            writer.writerow(["Dataset", "Language", "BLEU", "COMET"])
        writer.writerow([dataset, language, bleu, comet])

def write_translations_to_csv(dataset, language, sources, hypotheses, references):
    """Append source sentences, translations, and references to a separate CSV file."""
    with open(translations_csv_filename, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if file.tell() == 0:  # Write header only if the file is empty
            writer.writerow(["Dataset", "Language", "Source Sentence", "Translation", "Reference Sentence"])
        for source, hypothesis, reference in zip(sources, hypotheses, references):
            writer.writerow([dataset, language, source, hypothesis, reference])

for dataset_name, dataset_loader in DATASETS.items():
    print(f"🔹 Testing Gemini on {dataset_name}")
    
    # Get available language pairs from dataset loader
    try:
        language_pairs = dataset_loader("get_languages")
    except Exception as e:
        print(f"⚠️ Skipping {dataset_name}: Could not fetch language pairs ({e})")
        continue

    for lang_pair in language_pairs:
        print(f"🔄 Testing Gemini on {dataset_name} ({lang_pair})")
        
        try:
            sources, references = dataset_loader(lang_pair)
            if not sources:
                print(f"⚠️ Skipping {lang_pair} for {dataset_name}: No data available.")
                write_to_csv(dataset_name, lang_pair, "NA", "NA")
                continue
        except Exception as e:
            print(f"⚠️ Skipping {dataset_name} ({lang_pair}): Dataset loading error ({e})")
            write_to_csv(dataset_name, lang_pair, "NA", "NA")
            continue

        # Translate sentences with progress logging
        translations = []
        for i, src in enumerate(sources):
            print(f"🔄 Prompting Gemini: Translating sentence {i+1}/{len(sources)}: {src[:50]}...")
            try:
                prompt = f"Translate this sentence to {lang_pair.split('-')[-1]}: {src}"
                translation = translate_text(model, prompt)
                translations.append(translation)
            except Exception as e:
                print(f"⚠️ Translation error for {lang_pair}: {e}")
                translations.append("ERROR")

        # Compute metrics
        bleu = compute_bleu(references, translations)
        comet = compute_comet(references, translations, sources)
        print(f"📊 BLEU: {round(bleu, 2)}, COMET: {round(comet, 2)}")

        # Save to CSV
        write_to_csv(dataset_name, lang_pair, round(bleu, 2), round(comet, 2))
        write_translations_to_csv(dataset_name, lang_pair, sources, translations, references)

        print(f"✅ Saved results for Gemini on {dataset_name} ({lang_pair})")
