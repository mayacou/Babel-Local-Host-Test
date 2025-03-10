from google.cloud import translate_v2 as translate
import csv
from dotenv import load_dotenv
from helpers.evaluation import compute_bleu, compute_comet
from datasets_loader.load_wmt import load_wmt_data
from datasets_loader.load_tedTalk import load_tedTalk_data
from datasets_loader.load_europarl import load_europarl_data

load_dotenv()
client = translate.Client()

# Datasets to test
DATASETS = {
   "WMT": load_wmt_data,
   "TED": load_tedTalk_data,
   "Europarl": load_europarl_data
}

RESULTS_CSV = "data/GoogleCloud_test_results.csv"
TRANSLATIONS_CSV = "translation_results/GoogleCloud_translations.csv"

def translate(source_sentences, target_language, reference_sentences):
   res = []
   for sentence in source_sentences:
      result = client.translate(sentence, target_language=target_language)
      res.append(result["translatedText"])
   bleu = compute_bleu(reference_sentences, res)
   comet = compute_comet(reference_sentences, res, source_sentences)
   return res, bleu, comet
        
with open(RESULTS_CSV, mode="w", newline="") as file:
   writer = csv.writer(file)
   writer.writerow(["Dataset", "Language", "BLEU", "COMET"])

with open(TRANSLATIONS_CSV, mode="w", newline="") as trans_file:
   trans_writer = csv.writer(trans_file)
   trans_writer.writerow(["Dataset", "Language", "Source Sentence", "Translation", "Reference Sentence"])

   for dataset_name, dataset_loader in DATASETS.items():
      try:
         language_pairs = dataset_loader("get_languages")
         for language in language_pairs:
            source_sentences, reference_sentences = dataset_loader(language)
            translations, bleu, comet = translate(source_sentences, language, reference_sentences)
            writer.writerow([dataset_name, language, round(bleu, 2), round(comet, 2)])
            file.flush()
            
            for source, translation, reference in zip(source_sentences, translations, reference_sentences):
               trans_writer.writerow([dataset_name, language, source, translation, reference])
            trans_file.flush()
      except Exception as e:
         print(f"⚠️ Skipping {dataset_name} for {language}: {e}")
         writer.writerow([dataset_name, language, "Skipped", "Skipped"])
         file.flush()
