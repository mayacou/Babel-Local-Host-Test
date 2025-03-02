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
    "OPUS": load_opus_data,
    "Europarl": load_europarl_data
}

# Load Gemini model
print("🔄 Loading Gemini model...")
model = load_gemini()
print("✅ Gemini model loaded successfully!")

# Open CSV file to store results
csv_filename = "gemini_results.csv"
with open(csv_filename, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Dataset", "Language", "BLEU", "COMET"])

    for dataset_name, dataset_loader in DATASETS.items():
        print(f"🔹 Testing Gemini on {dataset_name}")
        
        # Get available language pairs from dataset loader
        language_pairs = dataset_loader("get_languages")
        
        for lang_pair in language_pairs:
            print(f"🔄 Testing Gemini on {dataset_name} ({lang_pair})")
            
            sources, references = dataset_loader(lang_pair)
            
            if not sources:
                print(f"⚠️ Skipping {lang_pair} for {dataset_name}: No data available.")
                continue

            # Translate sentences with progress logging
            translations = []
            for i, src in enumerate(sources):
                print(f"🔄 Prompting Gemini: Translating sentence {i+1}/{len(sources)}: {src[:50]}...")
                prompt = f"Translate this sentence to {lang_pair.split('-')[-1]}: {src}"
                translation = translate_text(model, prompt)
                translations.append(translation)
                print(f"✅ Gemini Translation: {translation}")
            
            # Compute metrics
            bleu = compute_bleu(references, translations)
            comet = compute_comet(references, translations, sources)
            print(f"📊 BLEU: {round(bleu, 2)}, COMET: {round(comet, 2)}")
            
            # Save to CSV
            writer.writerow([dataset_name, lang_pair, round(bleu, 2), round(comet, 2)])
            print(f"✅ Saved results for Gemini on {dataset_name} ({lang_pair})")

