import os
import csv
import shutil
import torch
from models.load_helsinki import load_model, translate_text
from helpers.evaluation import compute_bleu, compute_comet
from datasets_loader.load_wmt import load_wmt_data
from datasets_loader.load_tedTalk import load_tedTalk_data
from datasets_loader.load_europarl import load_europarl_data

# ✅ Define Datasets
DATASETS = {
    "Europarl": load_europarl_data,
    "WMT": load_wmt_data,
    "TED": load_tedTalk_data
}

# ✅ Define Models to Test
MODELS_TO_TEST = [
    "Helsinki-NLP/opus-mt-en-sq",
    "Helsinki-NLP/opus-mt-tc-big-en-gmq",
    "Helsinki-NLP/opus-mt-en-sla",
    "Helsinki-NLP/opus-mt-tc-big-en-bg",
    "Helsinki-NLP/opus-mt-tc-base-en-sh",
    "Helsinki-NLP/opus-mt-en-cs",
    "Helsinki-NLP/opus-mt-en-da",
    "Helsinki-NLP/opus-mt-en-nl",
    "Helsinki-NLP/opus-mt-tc-big-en-et",
    "Helsinki-NLP/opus-mt-tc-big-en-fi",
    "Helsinki-NLP/opus-mt-tc-big-en-fr",
    "Helsinki-NLP/opus-mt-en-de",
    "Helsinki-NLP/opus-mt-tc-big-en-el",
    "Helsinki-NLP/opus-mt-tc-big-en-hu",
    "Helsinki-NLP/opus-mt-en-is",
    "Helsinki-NLP/opus-mt-tc-big-en-it",
    "Helsinki-NLP/opus-mt-tc-big-en-lv",
    "Helsinki-NLP/opus-mt-tc-big-en-lt",
    "Helsinki-NLP/opus-mt-en-mk",
    "Helsinki-NLP/opus-mt-tc-big-en-pt",
    "Helsinki-NLP/opus-mt-tc-big-en-ro",
    "Helsinki-NLP/opus-mt-en-sk",
    "Helsinki-NLP/opus-mt-tc-big-en-es",
    "Helsinki-NLP/opus-mt-en-sv",
    "Helsinki-NLP/opus-mt-tc-big-en-tr",
]

# ✅ Language ID Mapping
LANG_ID_MAP = {
    "hr": ">>hrv<< ",
    "no": ">>nor<< ",
    "pl": ">>pol<< ",
    "sl": ">>slv<< ",
    "sv": ">>swe<< ",
    "is": ">>isl<< ",
    "da": ">>dan<< ",
}

# ✅ Cache Directory
CACHE_DIR = os.path.expanduser("~/.cache/huggingface")

# ✅ Define CSV Files
RESULTS_CSV = "data/scores/Helsinki_test_results.csv"
TRANSLATIONS_CSV = "data/translations/Helsinki_translations.csv"

# ✅ Clear Hugging Face Cache
def clear_huggingface_cache():
    if os.path.exists(CACHE_DIR):
        print(f"Clearing Hugging Face cache at {CACHE_DIR}...")
        shutil.rmtree(CACHE_DIR)
    else:
        print("No Hugging Face cache found to clear.")

# ✅ Translation Function
def translate(lang, model_name, source_sentences, reference_sentences):
    if not source_sentences or not reference_sentences:
        return -1, -1, []
    
    lang_id = ""
    target = model_name.split("-")[-1]
    if target in ["sla", "gmq", "sh"]:
        lang_id = LANG_ID_MAP.get(lang, "")
    
    model, tokenizer, device = load_model(model_name)
    translated_sentences = [translate_text(model, tokenizer, f"{lang_id}{sentence}", device) for sentence in source_sentences]
    bleu = compute_bleu(reference_sentences, translated_sentences)
    comet = compute_comet(reference_sentences, translated_sentences, source_sentences)
    
    return bleu, comet, translated_sentences

# ✅ Open CSV Files and Keep Them Open for the Entire Process
with open(RESULTS_CSV, mode="w", newline="") as file, open(TRANSLATIONS_CSV, mode="w", newline="") as trans_file:
    writer = csv.writer(file)
    trans_writer = csv.writer(trans_file)
    
    # ✅ Write CSV Headers
    writer.writerow(["Model Name", "Dataset", "Language", "BLEU", "COMET"])
    trans_writer.writerow(["Model Name", "Dataset", "Language", "Source Sentence", "Translation", "Reference Sentence"])
    
    # ✅ Iterate Over Models
    for model in MODELS_TO_TEST:
        lang_pair = [model.split("-")[-1]]
        if lang_pair[0] == "sla":
            lang_pair = ["hr", "sl", "pl"]
        elif lang_pair[0] == "sh":
            lang_pair = ["hr"]
        elif lang_pair[0] == "gmq":
            lang_pair = ["no", "sv", "is", "da"]
        
        # ✅ Iterate Over Languages
        for lang in lang_pair:
            # ✅ Iterate Over Datasets
            for dataset_name, dataset_loader in DATASETS.items():
                try:
                    source_sentences, reference_sentences = dataset_loader(lang)
                    
                    if not source_sentences or not reference_sentences:
                        raise ValueError(f"No dataset found for {lang} in {dataset_name}")
                    
                    # ✅ Perform Translation
                    bleu, comet, translations = translate(lang, model, source_sentences, reference_sentences)
                    
                    # ✅ Save Results to CSV
                    writer.writerow([model, dataset_name, lang, round(bleu, 2), round(comet, 2)])
                    file.flush()  # ✅ Ensure immediate writing to the file
                    
                    # ✅ Save Translations to CSV
                    for source, translation, reference in zip(source_sentences, translations, reference_sentences):
                        trans_writer.writerow([model, dataset_name, lang, source, translation, reference])
                    trans_file.flush()  # ✅ Ensure immediate writing to the file
                    
                except Exception as e:
                    print(f"⚠️ Skipping {dataset_name} for {lang}: {e}")
                    writer.writerow([model, dataset_name, lang, "Skipped", "Skipped"])
                    file.flush()
            
        # ✅ Clear Cache After Each Model to Prevent Disk Space Issues
        clear_huggingface_cache()
