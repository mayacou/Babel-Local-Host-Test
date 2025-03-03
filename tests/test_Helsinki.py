import csv
from models.load_helsinki import load_model, translate_text
from helpers.evaluation import compute_bleu, compute_comet
from datasets_loader.load_wmt import load_wmt_data
from datasets_loader.load_ted import load_ted_data
from datasets_loader.load_europarl import load_europarl_data


DATASETS = {
    "Europarl": load_europarl_data,
    #"WMT": load_wmt_data,
    "TED": load_ted_data
}

# need to update to convert the Helsinki last language to WMT's format
HELSINKI_TO_WMT = {
    
}

MODELS_TO_TEST = [
    "Helsinki-NLP/opus-mt-en-sq",  # English → Albanian (NO EUROPARL DATASET)
    "Helsinki-NLP/opus-mt-en-bg",  # English → Bulgarian
    "Helsinki-NLP/opus-mt-tc-big-en-bg",  # English → Bulgarian
    # "Helsinki-NLP/opus-mt-tc-base-en-sh",  # English → Croatian (NO EUROPARL DATASET, serbo-croatian model, handles serbian, croatian, serbo-croatian, bosnian)
    "Helsinki-NLP/opus-mt-en-cs",  # English → Czech
    "Helsinki-NLP/opus-mt-en-da",  # English → Danish
    "Helsinki-NLP/opus-mt-en-nl",  # English → Dutch
    "Helsinki-NLP/opus-mt-en-et",  # English → Estonian
    "Helsinki-NLP/opus-mt-tc-big-en-et",  # English → Estonian
    "Helsinki-NLP/opus-mt-en-fi",  # English → Finnish
    "Helsinki-NLP/opus-mt-tc-big-en-fi",  # English → Finnish
    "Helsinki-NLP/opus-mt-en-fr",  # English → French
    "Helsinki-NLP/opus-mt-tc-big-en-fr",  # English → French
    "Helsinki-NLP/opus-mt-en-de",  # English → German
    "Helsinki-NLP/opus-mt-en-el",  # English → Greek 
    "Helsinki-NLP/opus-mt-tc-big-en-el",  # English → Greek
    "Helsinki-NLP/opus-mt-en-hu",  # English → Hungarian
    "Helsinki-NLP/opus-mt-tc-big-en-hu",  # English → Hungarian
    "Helsinki-NLP/opus-mt-en-is", # English → Icelandic (NO EUROPARL DATASET)
    "Helsinki-NLP/opus-mt-en-it",  # English → Italian
    "Helsinki-NLP/opus-mt-tc-big-en-it", # English → Italian
    "Helsinki-NLP/opus-mt-tc-big-en-lv",  # English → Latvian 
    "Helsinki-NLP/opus-mt-tc-big-en-lt",  # English → Lithuanian
    "Helsinki-NLP/opus-mt-en-mk",  # English → Macedonian (NO EUROPARL DATASET)
    # "Helsinki-NLP/opus-mt-tc-big-en-gmq",  # English → Norwegian (NO EUROPARL DATASET, NORTH GERMANIC LANGUAGES)
    # "Helsinki-NLP/opus-mt-en-sla",  # English → Polish (slavic language multi purpose model)
    "Helsinki-NLP/opus-mt-tc-big-en-pt",  # English → Portuguese 
    "Helsinki-NLP/opus-mt-en-ro",  # English → Romanian 
    "Helsinki-NLP/opus-mt-tc-big-en-ro",  # English → Romanian 
    "Helsinki-NLP/opus-mt-en-sk",  # English → Slovak
    # "Helsinki-NLP/opus-mt-en-sla",  # English → Slovenian
    "Helsinki-NLP/opus-mt-en-es",  # English → Spanish
    "Helsinki-NLP/opus-mt-tc-big-en-es",  # English → Spanish (Helsinki-NLP/opus-mt-tc-big-en-cat_oci_spa for catalan if necessary)
    "Helsinki-NLP/opus-mt-en-sv",  # English → Swedish
    "Helsinki-NLP/opus-mt-tc-big-en-tr",   # English → Turkish (NO EUROPARL DATASET)
]

def translate(model_name, source_sentences, reference_sentences):
    if not source_sentences or not reference_sentences:
        return -1, -1
        
    # Helsinki Test
    model, tokenizer = load_model(model_name)
    translated_sentences = [translate_text(model, tokenizer, sentence) for sentence in source_sentences]
    bleu = compute_bleu(reference_sentences, translated_sentences)
    comet = compute_comet(reference_sentences, translated_sentences, source_sentences) * 100

    return bleu, comet
    
# Note: Helsinki only tests on ONE language, so no need to loop through language pairs.
csv_filename = "Helsinki_test_results.csv"
with open(csv_filename, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Dataset", "Language", "BLEU", "COMET"])
    for model in MODELS_TO_TEST:
        lang_pair = model.split("-")[-1]
        for dataset_name, dataset_loader in DATASETS.items():
            try:
                if (dataset_name == "WMT"):
                    target_lang = HELSINKI_TO_WMT[target_lang]
                source_sentences, reference_sentences = dataset_loader(lang_pair)
                bleu, comet = translate(model, source_sentences, reference_sentences)
                writer.writerow([dataset_name, lang_pair, round(bleu, 2), round(comet, 2)])
                file.flush()
            except Exception as e:
                print(f"⚠️ Skipping {dataset_name} for {lang_pair}: {e}")
                writer.writerow([dataset_name, lang_pair, "Skipped", "Skipped"])
                file.flush()
            