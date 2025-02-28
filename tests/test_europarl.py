import json
import pytest
import os
import asyncio
from datasets import load_dataset
from helpers.model_loader import load_model, translate_text
from helpers.evaluation import compute_bleu, compute_comet
from GoogleTranslate import google_translate
from GPTtest import translate_with_chatgpt

BATCH_SIZE = 5
PADDING = 90545
TEST_GPT = True
# finds data directory outside of tests folder and gets path to json to store test data
DATA_DIR = os.path.join(os.path.dirname(os.getcwd()), "data")
RESULTS_JSON = os.path.join(DATA_DIR, "europarl_test_results.json")

MODELS_TO_TEST = [
    # "Helsinki-NLP/opus-mt-en-sq",  # English → Albanian (NO EUROPARL DATASET)
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
    # "Helsinki-NLP/opus-mt-en-is", # English → Icelandic (NO EUROPARL DATASET)
    "Helsinki-NLP/opus-mt-en-it",  # English → Italian
    "Helsinki-NLP/opus-mt-tc-big-en-it", # English → Italian
    "Helsinki-NLP/opus-mt-tc-big-en-lv",  # English → Latvian 
    "Helsinki-NLP/opus-mt-tc-big-en-lt",  # English → Lithuanian
    # "Helsinki-NLP/opus-mt-en-mk",  # English → Macedonian (NO EUROPARL DATASET)
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
    # "Helsinki-NLP/opus-mt-tc-big-en-tr",   # English → Turkish (NO EUROPARL DATASET)
]

def get_bilingual_dataset(language_pair):
    try:
        dataset = load_dataset("Helsinki-NLP/europarl", language_pair, split="train")
        
        source_lang, target_lang = language_pair.split("-")
        
        # Ensure English is always the source language
        if source_lang != "en":
            source_lang, target_lang = target_lang, source_lang
        
        dataset = dataset.shuffle(seed=42)
        source_sentences = []
        reference_sentences = []
        for item in dataset.select(range(BATCH_SIZE)):  
            if "translation" in item and source_lang in item["translation"] and target_lang in item["translation"]:
                source_sentences.append(item["translation"][source_lang])
                reference_sentences.append(item["translation"][target_lang])
            
        return source_sentences, reference_sentences, target_lang
    except Exception as e:
        print(f"Error loading dataset for {language_pair}: {e}")
        return [], [], ""

@pytest.mark.parametrize("model_name", MODELS_TO_TEST)
def test_translate(model_name):
    lang_pair = model_name.split("-")[-2] + "-" + model_name.split("-")[-1]
    lang_pair = "-".join(sorted(lang_pair.split("-")))
    source_sentences, reference_sentences, target = get_bilingual_dataset(lang_pair)
        
    if not source_sentences or not reference_sentences:
        pytest.skip(f"Skipping model {model_name}: No data for {lang_pair}")
    
    # Load existing results or initialize as an empty list
    if os.path.exists(RESULTS_JSON):
        with open(RESULTS_JSON, "r") as f:
            try:
                results = json.load(f)
            except json.JSONDecodeError:
                results = []
    else:
        results = []
        
    # Helsinki Test
    model, tokenizer = load_model(model_name)
    translated_sentences = [translate_text(model, tokenizer, sentence) for sentence in source_sentences]
    bleu = compute_bleu(reference_sentences, translated_sentences)
    comet = compute_comet(reference_sentences, translated_sentences, source_sentences) * 100
    results.append({"model": model_name, "Langauge": target, "BLEU": round(bleu, 2), "COMET": round(comet, 2)})
    
    
    # Google Translate test
    translated_sentences = []
    translated_sentences = asyncio.run(google_translate(source_sentences, target_language=target))
    bleu = compute_bleu(reference_sentences, translated_sentences)
    comet = compute_comet(reference_sentences, translated_sentences, source_sentences) * 100
    results.append({"model": "Google Translate", "Language": target, "bleu": round(bleu, 2), "comet": round(comet, 2)})
    
    
    # Chat GPT test (only run when actaully testing - costs money)
    if (TEST_GPT == True): 
        translated_sentences = []
        for i in range(0, len(source_sentences), 3):
            j = i + 3
            if i + 3 >= len(source_sentences): j = len(source_sentences)
            batch = source_sentences[i:j]
            batch_translations = translate_with_chatgpt(batch, target_language=target)
            translated_sentences.extend(batch_translations)
        bleu = compute_bleu(reference_sentences, translated_sentences)
        comet = compute_comet(reference_sentences, translated_sentences, source_sentences) * 100
        results.append({"model": "ChatGPT-4o-mini", "Language": target, "bleu": round(bleu, 2), "comet": round(comet, 2)})



    # Save the updated results list back to the file
    with open(RESULTS_JSON, "w") as f:
        json.dump(results, f, indent=4)
    
    # **Assertions for pytest**
    assert bleu > 0, f"BLEU score is too low for model {model_name}"
    assert comet is not None, f"COMET score is None for model {model_name}"