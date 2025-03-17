import csv
import torch
from models.load_towerinstruct import load_towerinstruct, translate_text
from helpers.evaluation import compute_bleu, compute_comet
from datasets_loader.load_wmt import load_wmt_data
from datasets_loader.load_tedTalk import load_tedTalk_data
from datasets_loader.load_opus import load_opus_data
from datasets_loader.load_europarl import load_europarl_data
import os

# Ensure the directories exist
os.makedirs("translation_results", exist_ok=True)
os.makedirs("data", exist_ok=True)

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
    model, tokenizer, device = load_towerinstruct(model_version)
    print(f"✅ {model_name} loaded successfully!")

    # Define file paths
    csv_filename = f"data/towerinstruct_{model_version}b_results.csv"
    translations_csv_filename = f"translation_results/towerinstruct_{model_version}b_translations.csv"

    # ✅ Open CSV files once for the entire loop
    with open(csv_filename, mode="w", newline="") as file, open(translations_csv_filename, mode="w", newline="") as trans_file:
        writer = csv.writer(file)
        trans_writer = csv.writer(trans_file)

        # ✅ Write headers if files are empty
        if os.stat(csv_filename).st_size == 0:
            writer.writerow(["Dataset", "Language", "BLEU", "COMET"])

        if os.stat(translations_csv_filename).st_size == 0:
            trans_writer.writerow(["Dataset", "Language", "Source Sentence", "Translation", "Reference Sentence"])

        for dataset_name, dataset_loader in DATASETS.items():
            print(f"🔹 Testing {model_name} on {dataset_name}")

            # ✅ Handle dataset errors
            try:
                language_pairs = dataset_loader("get_languages")
            except Exception as e:
                print(f"⚠️ Error loading languages for {dataset_name}: {e}")
                continue

            for lang_pair in language_pairs:
                print(f"🔄 Testing {model_name} on {dataset_name} ({lang_pair})")

                try:
                    sources, references = dataset_loader(lang_pair)
                except Exception as e:
                    print(f"⚠️ Error loading data for {lang_pair}: {e}")
                    writer.writerow([dataset_name, lang_pair, "NA", "NA"])
                    continue

                if not sources:
                    print(f"⚠️ Skipping {lang_pair} for {dataset_name}: No data available.")
                    writer.writerow([dataset_name, lang_pair, "NA", "NA"])
                    continue

                # Translate sentences with progress logging
                translations = []
                for i, src in enumerate(sources):
                    print(f"🔄 Translating sentence {i+1}/{len(sources)}: {src[:50]}...")
                    try:
                        translation = translate_text(model, tokenizer, src, device)
                        translations.append(translation)
                        print(f"✅ Translated: {translation[:50]}")
                        trans_writer.writerow([dataset_name, lang_pair, src, translation, references[i]])
                    except Exception as e:
                        print(f"⚠️ Translation error: {e}")
                        translations.append("ERROR")
                        trans_writer.writerow([dataset_name, lang_pair, src, "ERROR", references[i]])

                trans_file.flush()

                # Compute metrics safely
                try:
                    bleu = compute_bleu(references, translations)
                    comet = compute_comet(references, translations, sources)
                    print(f"📊 BLEU: {round(bleu, 2)}, COMET: {round(comet, 2)}")
                except Exception as e:
                    print(f"⚠️ Error computing BLEU/COMET: {e}")
                    bleu, comet = "ERROR", "ERROR"

                # ✅ Save to CSV
                writer.writerow([dataset_name, lang_pair, round(bleu, 2) if bleu != "ERROR" else "NA",
                                 round(comet, 2) if comet != "ERROR" else "NA"])
                file.flush()

                print(f"✅ Saved results for {model_name} on {dataset_name} ({lang_pair})")
