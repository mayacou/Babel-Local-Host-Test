import csv 
from models.load_towerinstruct import load_towerinstruct, translate_text
from helpers.evaluation import compute_bleu, compute_comet
from datasets_loader.load_wmt import load_wmt_data
from datasets_loader.load_tedTalk import load_tedTalk_data
from datasets_loader.load_opus import load_opus_data
from datasets_loader.load_europarl import load_europarl_data

# Datasets to test
DATASETS = {
    "WMT": load_wmt_data,
    "TED": load_tedTalk_data,
    #"OPUS": load_opus_data,
    "Europarl": load_europarl_data
}

for model_version in [7, 13]:
    model_name = f"TowerInstruct-{model_version}B"
    print(f"🔄 Loading {model_name} model...")
    model, tokenizer = load_towerinstruct(model_version)
    print(f"✅ {model_name} loaded successfully!")
    
    # Open separate CSV files for results and translations
    csv_filename = f"data/towerinstruct_{model_version}b_results.csv"
    translations_csv_filename = f"translation_results/towerinstruct_{model_version}b_translations.csv"
    
    with open(csv_filename, mode="w", newline="") as file, open(translations_csv_filename, mode="w", newline="") as trans_file:
        writer = csv.writer(file)
        trans_writer = csv.writer(trans_file)
        writer.writerow(["Dataset", "Language", "BLEU", "COMET"])
        trans_writer.writerow(["Dataset", "Language", "Source Sentence", "Translation", "Reference Sentence"])

        for dataset_name, dataset_loader in DATASETS.items():
            print(f"🔹 Testing {model_name} on {dataset_name}")
            
            # Get available language pairs from dataset loader
            language_pairs = dataset_loader("get_languages")
            
            for lang_pair in language_pairs:
                print(f"🔄 Testing {model_name} on {dataset_name} ({lang_pair})")
                
                sources, references = dataset_loader(lang_pair)
                
                if not sources:
                    print(f"⚠️ Skipping {lang_pair} for {dataset_name}: No data available.")
                    writer.writerow([dataset_name, lang_pair, "NA", "NA"])
                    continue

                # Translate sentences with progress logging
                translations = []
                for i, src in enumerate(sources):
                    print(f"🔄 Translating sentence {i+1}/{len(sources)}: {src[:50]}...")
                    translation = translate_text(model, tokenizer, src)
                    translations.append(translation)
                    print(f"✅ Translated: {translation[:50]}")
                    trans_writer.writerow([dataset_name, lang_pair, src, translation, references[i]])
                    trans_file.flush()
                
                # Compute metrics
                bleu = compute_bleu(references, translations)
                comet = compute_comet(references, translations, sources)
                print(f"📊 BLEU: {round(bleu, 2)}, COMET: {round(comet, 2)}")
                
                # Save to CSV
                writer.writerow([dataset_name, lang_pair, round(bleu, 2), round(comet, 2)])
                print(f"✅ Saved results for {model_name} on {dataset_name} ({lang_pair})")



