import json
import pytest
import os
from datasets import load_dataset
from helpers.model_loader import load_model, translate_text
from helpers.evaluation import compute_bleu, compute_comet

RESULTS_JSON = "data/europarl_test_results.json"
BATCH_SIZE = 5
PADDING = 90545

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
    # "Helsinki-NLP/opus-mt-tc-big-en-gmq", # English → Icelandic (NO EUROPARL DATASET, gmq covers north germanic languages including icelandic - also covers norwiegan, swedish, danish and faroese)
    "Helsinki-NLP/opus-mt-en-it",  # English → Italian
    "Helsinki-NLP/opus-mt-tc-big-en-it", # English → Italian
    "Helsinki-NLP/opus-mt-tc-big-en-lv",  # English → Latvian (Neural machine translation model - made in 2022)
    "Helsinki-NLP/opus-mt-tc-big-en-lt",  # English → Lithuanian (Neural machine translation model - made in 2022)
    # "Helsinki-NLP/opus-mt-en-mk",  # English → Macedonian (NO EUROPARL DATASET)
    # "Helsinki-NLP/opus-mt-es-NORWAY",  # English → Norwegian (NO EUROPARL DATASET)
    # "Helsinki-NLP/opus-mt-en-sla",  # English → Polish (slavic language multi purpose model)
    "Helsinki-NLP/opus-mt-tc-big-en-pt",  # English → Portuguese (Neural machine translation model - made in 2022)
    "Helsinki-NLP/opus-mt-en-ro",  # English → Romanian 
    "Helsinki-NLP/opus-mt-tc-big-en-ro",  # English → Romanian 
    "Helsinki-NLP/opus-mt-en-sk",  # English → Slovak
    # "Helsinki-NLP/opus-mt-en-sla",  # English → Slovenian (slavic language multi purpose model)
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
            
        return source_sentences, reference_sentences
    except Exception as e:
        print(f"Error loading dataset for {language_pair}: {e}")
        return [], []

@pytest.mark.parametrize("model_name", MODELS_TO_TEST)
def test_translate(model_name):
    print(f"Current Model Name: {model_name}")
        
    lang_pair = model_name.split("-")[-2] + "-" + model_name.split("-")[-1]
    lang_pair = "-".join(sorted(lang_pair.split("-")))
    source_sentences, reference_sentences = get_bilingual_dataset(lang_pair)
        
    if not source_sentences or not reference_sentences:
        pytest.skip(f"Skipping model {model_name}: No data for {lang_pair}")
        
    # Load translation model (Example: English to French)
    model, tokenizer = load_model(model_name)
    translated_sentences = [translate_text(model, tokenizer, sentence) for sentence in source_sentences]
    
    # Debugging print statements
    print("\n--- Translation Debugging Output ---")
    for src, hyp, ref in zip(source_sentences, translated_sentences, reference_sentences):
        print(f"🔹 Source: {src}")
        print(f"🔹 Hypothesis: {hyp}")
        print(f"🔹 Reference: {ref}")
        print("----")
    
    # Compute BLEU and COMET scores
    bleu = compute_bleu(reference_sentences, translated_sentences)
    comet = compute_comet(reference_sentences, translated_sentences, source_sentences)

    # Save results
    result = {
        "model": model_name,
        "BLEU": round(bleu, 2),
        "COMET": round(comet, 2),
    }
    
    # Load existing results or initialize as an empty list
    if os.path.exists(RESULTS_JSON):
        with open(RESULTS_JSON, "r") as f:
            try:
                results = json.load(f)
            except json.JSONDecodeError:
                results = []
    else:
        results = []

    # Append the new result
    results.append(result)

    # Save the updated results list back to the file
    with open(RESULTS_JSON, "w") as f:
        json.dump(results, f, indent=4)

    
    # **Assertions for pytest**
    assert bleu > 0, f"BLEU score is too low for model {model_name}"
    assert comet is not None, f"COMET score is None for model {model_name}"
