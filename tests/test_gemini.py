import csv
from models.load_gemini import load_gemini, translate_text
from helpers.evaluation import compute_bleu, compute_comet
from datasets_loader.load_wmt import load_wmt_data
from datasets_loader.load_ted import load_ted_data
from datasets_loader.load_opus import load_opus_data
from datasets_loader.load_europarl import load_europarl_data

# Datasets to test
DATASETS = {
    "WMT": load_wmt_data,
    "TED": load_ted_data,
    #"OPUS": load_opus_data,
    "Europarl": load_europarl_data
}

# Load Gemini model
print("🔄 Loading Gemini model...")
model = load_gemini()
print("✅ Gemini model loaded successfully!")

# Open CSV files to store results
csv_filename = "data/gemini_results.csv"
translations_csv_filename = "translation_results/gemini_translations.csv"

with open(csv_filename, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Dataset", "Language", "BLEU", "COMET"])

with open(translations_csv_filename, mode="w", newline="") as trans_file:
    trans_writer = csv.writer(trans_file)
    trans_writer.writerow(["Dataset", "Language", "Source Sentence", "Translation", "Reference Sentence"])

    for dataset_name, dataset_loader in DATASETS.items():
        print(f"🔹 Testing Gemini on {dataset_name}")
        
        # Get available language pairs from dataset loader
        language_pairs = dataset_loader("get_languages")
        
        for lang_pair in language_pairs:
            print(f"🔄 Testing Gemini on {dataset_name} ({lang_pair})")
            
            sources, references = dataset_loader(lang_pair)
            
            if not sources:
                print(f"⚠️ Skipping {lang_pair} for {dataset_name}: No data available.")
                writer.writerow([dataset_name, lang_pair, "NA", "NA"])
                continue

            # Translate sentences with progress logging
            translations = []
            for i, src in enumerate(sources):
                print(f"🔄 Prompting Gemini: Translating sentence {i+1}/{len(sources)}: {src[:50]}...")
                prompt = f"Translate this sentence to {lang_pair.split('-')[-1]}: {src}"
                translation = translate_text(model, prompt)
                translations.append(translation)
                trans_writer.writerow([dataset_name, lang_pair, src, translation, references[i]])
                trans_file.flush()
            
            # Compute metrics
            bleu = compute_bleu(references, translations)
            comet = compute_comet(references, translations, sources)
            print(f"📊 BLEU: {round(bleu, 2)}, COMET: {round(comet, 2)}")
            
            # Save to CSV
            writer.writerow([dataset_name, lang_pair, round(bleu, 2), round(comet, 2)])
            print(f"✅ Saved results for Gemini on {dataset_name} ({lang_pair})")


