from google.cloud import translate_v2 as translate
import csv
from dotenv import load_dotenv
from helpers.evaluation import compute_bleu, compute_comet
from datasets_loader.load_wmt import load_wmt_data
from datasets_loader.load_ted import load_ted_data
from datasets_loader.load_europarl import load_europarl_data

load_dotenv()

client = translate.Client()

# Datasets to test
DATASETS = {
   #"WMT": load_wmt_data,
   #"TED": load_ted_data,
   "Europarl": load_europarl_data
}

def translate(source_sentences, target_language, reference_sentences):
   res = []
   for sentence in source_sentences:
      result = client.translate(sentence, target_language=target_language)
      res.append(result["translatedText"])
   bleu = compute_bleu(reference_sentences, res)
   comet = compute_comet(reference_sentences, res, source_sentences) * 100
   return bleu, comet
        
csv_filename = "GoogleCloud_test_results.csv"
with open(csv_filename, mode="w", newline="") as file:
   writer = csv.writer(file)
   writer.writerow(["Dataset", "Language", "BLEU", "COMET"])
   for dataset_name, dataset_loader in DATASETS.items():
      try:
         language_pairs = dataset_loader("get_languages")
         for language in language_pairs:
            source_sentences, reference_sentences = dataset_loader(language)
            bleu, comet = translate(source_sentences, language, reference_sentences)
            writer.writerow([dataset_name, language, round(bleu, 2), round(comet, 2)])
            file.flush()
      except Exception as e:
         print(f"⚠️ Skipping {dataset_name} for {language}: {e}")
         writer.writerow([dataset_name, language, "Skipped", "Skipped"])
         file.flush()