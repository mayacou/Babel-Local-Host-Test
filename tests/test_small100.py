import os
import csv
from helpers.evaluation import compute_bleu, compute_comet
from models.load_small100 import load_small100
from datasets_loader.load_tedTalk import load_tedTalk_data
from datasets_loader.load_europarl import load_europarl_data
from datasets_loader.load_wmt import load_wmt_data

# Setup
os.makedirs("data", exist_ok=True)
os.makedirs("translation_results", exist_ok=True)
RESULTS_CSV = "data/small100_results(maya).csv"
TRANSLATIONS_CSV = "translation_results/small100_translations(maya).csv"

DATASETS = {
    "TED": load_tedTalk_data,
    "Europarl": load_europarl_data,
    "WMT": load_wmt_data
}

# Track already completed entries
completed = set()
if os.path.exists(RESULTS_CSV):
    with open(RESULTS_CSV, mode="r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header
        for row in reader:
            if len(row) >= 2:
                completed.add((row[0], row[1]))  # (dataset_name, language)

with open(RESULTS_CSV, mode="a", newline="", encoding="utf-8") as results_file, \
     open(TRANSLATIONS_CSV, mode="a", newline="", encoding="utf-8") as trans_file:

    results_writer = csv.writer(results_file)
    trans_writer = csv.writer(trans_file)

    # Write headers if files are empty
    if os.stat(RESULTS_CSV).st_size == 0:
        results_writer.writerow(["Dataset", "Language", "BLEU", "COMET"])

    if os.stat(TRANSLATIONS_CSV).st_size == 0:
        trans_writer.writerow(["Dataset", "Language", "Source Sentence", "Translation", "Reference Sentence"])

    for dataset_name, dataset_loader in DATASETS.items():
        print(f"📚 Testing on {dataset_name}")
        languages = dataset_loader("get_languages")

        for lang in languages:
            if (dataset_name, lang) in completed:
                print(f"⏩ Skipping already processed: {dataset_name} | {lang}")
                continue

            try:
                sources, references = dataset_loader(lang)
                if not sources:
                    print(f"⚠️ No data for {lang} in {dataset_name}")
                    results_writer.writerow([dataset_name, lang, "NA", "NA"])
                    continue

                model, tokenizer, device = load_small100(lang)

                # Translate
                translations = []
                for sentence in sources:
                    inputs = tokenizer(sentence, return_tensors="pt").to(device)
                    output = model.generate(**inputs, num_beams=10, max_length=256, early_stopping=True)
                    translation = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
                    translations.append(translation)

                # Evaluate
                bleu = compute_bleu(references, translations)
                comet = compute_comet(references, translations, sources)
                print(f"✅ {dataset_name} | {lang} -> BLEU: {bleu}, COMET: {comet}")

                # Save scores
                results_writer.writerow([dataset_name, lang, round(bleu, 2), round(comet, 2)])
                results_file.flush()

                # Save translations
                for src, hyp, ref in zip(sources, translations, references):
                    trans_writer.writerow([dataset_name, lang, src, hyp, ref])
                trans_file.flush()

            except Exception as e:
                print(f"❌ Error for {dataset_name} | {lang}: {e}")
                results_writer.writerow([dataset_name, lang, "Error", "Error"])
                results_file.flush()



