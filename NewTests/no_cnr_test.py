import csv
from models.load_helsinki import load_model, translate_text
from helpers.evaluation import compute_bleu, compute_comet
from datasets_loader.load_wmt import load_wmt_data
from datasets_loader.load_tedTalk import load_tedTalk_data
from datasets_loader.load_europarl import load_europarl_data

DATASETS = {
   #"Europarl": load_europarl_data,
   "WMT": load_wmt_data, # has no
   "TED": load_tedTalk_data # has no as nb 
}

MODELS_TO_TEST = [
   "Helsinki-NLP/opus-mt-tc-big-en-gmq", # tests norweigan as no, also icelandic as is
   "Helsinki-NLP/opus-mt-tc-base-en-sh", # tests montenegrin as cnr - need dataset
]

# Language ID Mapping
LANG_ID_MAP = {
   "no": ">>nor<< ",
   "pl": ">>pol<< ",
   "sv": ">>swe<< ",
   "is": ">>isl<< ",
   "da": ">>dan<< ",
   "cnr": ">>cnr<<",
}

RESULTS_CSV = "data/scores/no_cnr_results.csv"
TRANSLATIONS_CSV = "data/translations/no_cnr_translations.csv"

# Translation Function
def translate(lang, model_name, source_sentences, reference_sentences):
   if not source_sentences or not reference_sentences:
      return -1, -1, []
   lang_id = LANG_ID_MAP.get(lang, "")
    
   model, tokenizer, device = load_model(model_name)
   translated_sentences = [translate_text(model, tokenizer, f"{lang_id}{sentence}", device) for sentence in source_sentences]
   bleu = compute_bleu(reference_sentences, translated_sentences)
   comet = compute_comet(reference_sentences, translated_sentences, source_sentences)
    
   return bleu, comet, translated_sentences

# Open CSV Files and Keep Them Open for the Entire Process
with open(RESULTS_CSV, mode="w", newline="") as file, open(TRANSLATIONS_CSV, mode="w", newline="") as trans_file:
   writer = csv.writer(file)
   trans_writer = csv.writer(trans_file)
    
   # CSV Headers
   writer.writerow(["Model Name", "Dataset", "Language", "BLEU", "COMET"])
   trans_writer.writerow(["Model Name", "Dataset", "Language", "Source Sentence", "Translation", "Reference Sentence"])

   i = 1
   for model in MODELS_TO_TEST:
      if i == 1:
         lang_pair = ["no", "sv", "is", "da"]
      elif i == 2:
         lang_pair = ["cnr"]
      
      # Iterate Over Languages
      for lang in lang_pair:
         # Iterate Over Datasets
         for dataset_name, dataset_loader in DATASETS.items():
            try:
               source_sentences, reference_sentences = dataset_loader(lang)
                    
               if not source_sentences or not reference_sentences: raise ValueError(f"No dataset found for {lang} in {dataset_name}")
                    
               # Perform Translation
               bleu, comet, translations = translate(lang, model, source_sentences, reference_sentences)
                    
               # Save Results to CSV
               writer.writerow([model, dataset_name, lang, round(bleu, 2), round(comet, 2)])
               file.flush()  # ✅ Ensure immediate writing to the file
                    
               # Save Translations to CSV
               for source, translation, reference in zip(source_sentences, translations, reference_sentences):
                  trans_writer.writerow([model, dataset_name, lang, source, translation, reference])
               trans_file.flush()  # ✅ Ensure immediate writing to the file
            except Exception as e:
               print(f"⚠️ Skipping {dataset_name} for {lang}: {e}")
               writer.writerow([model, dataset_name, lang, "Skipped", "Skipped"])
               file.flush()
      i+=1
      
