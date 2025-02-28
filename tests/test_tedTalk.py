import os
import json
import pytest
import re
from datasets import load_dataset
from transformers import MarianMTModel, MarianTokenizer
from sacrebleu import corpus_bleu

# Define the path for results inside the 'data' folder
RESULTS_FILE = "data/ted_talks_test_results.json"
os.makedirs("data", exist_ok=True)

# Dictionary of models to test
MODELS_TO_TEST = {
    "English-Albanian": "Helsinki-NLP/opus-mt-en-sq",
    "English-Bulgarian": "Helsinki-NLP/opus-mt-en-bg",
    #"English-Croatian": "Helsinki-NLP/opus-mt-tc-base-en-sh",
    "English-Czech": "Helsinki-NLP/opus-mt-en-cs",
    "English-Danish": "Helsinki-NLP/opus-mt-en-da",
    "English-Dutch": "Helsinki-NLP/opus-mt-en-nl",
    "English-Estonian": "Helsinki-NLP/opus-mt-en-et",
    "English-Finnish": "Helsinki-NLP/opus-mt-en-fi",
    "English-French": "Helsinki-NLP/opus-mt-en-fr",
    "English-German": "Helsinki-NLP/opus-mt-en-de",
    "English-Greek": "Helsinki-NLP/opus-mt-en-el",
    "English-Hungarian": "Helsinki-NLP/opus-mt-en-hu",
    "English-Italian": "Helsinki-NLP/opus-mt-en-it",
    "English-Icelandic": "Helsinki-NLP/opus-mt-en-is",
    "English-Latvian": "Helsinki-NLP/opus-mt-tc-big-en-lv",
    "English-Lithuanian": "Helsinki-NLP/opus-mt-tc-big-en-lt",
    "English-Macedonian": "Helsinki-NLP/opus-mt-en-mk",
    #"English-Polish": "Helsinki-NLP/opus-mt-en-sla",
    "English-Portuguese": "Helsinki-NLP/opus-mt-tc-big-en-pt",
    "English-Romanian": "Helsinki-NLP/opus-mt-en-ro",
    "English-Slovak": "Helsinki-NLP/opus-mt-en-sk",
    #"English-Slovenian": "Helsinki-NLP/opus-mt-en-sla",
    "English-Spanish": "Helsinki-NLP/opus-mt-en-es",
    "English-Swedish": "Helsinki-NLP/opus-mt-en-sv",
    "English-Turkish": "Helsinki-NLP/opus-mt-tc-big-en-tr",
}

def extract_lang_pair(model_name):
    """
    Extracts source and target language pairs from a given model name.
    
    Handles models with additional tags like "-tc-big", "-tc-base", etc.

    Example Inputs:
        "Helsinki-NLP/opus-mt-en-fr" -> ("en", "fr")
        "Helsinki-NLP/opus-mt-tc-base-en-sh" -> ("en", "sh")
        "Helsinki-NLP/opus-mt-tc-big-en-XX" -> ("en", "XX")
    
    Returns:
        (str, str): Source language, Target language
    """
    # Match both standard and complex model naming formats
    match = re.search(r"opus-mt(?:-[a-z]+)*-(\w+)-(\w+)", model_name)
    
    if match:
        source_lang, target_lang = match.groups()
        return source_lang, target_lang
    else:
        raise ValueError(f"Could not extract language pair from model: {model_name}")

@pytest.mark.parametrize("lang_pair,model_name", MODELS_TO_TEST.items())
def test_translation_quality(lang_pair, model_name):
    """Test translation models using WMT data and log results in JSON."""
    
    print(f"🔹 Testing model: {model_name} ({lang_pair})")  # Debugging output

    # Load model & tokenizer
    model, tokenizer = load_model(model_name)

    # Load WMT dataset
    sources, references = load_wmt_data(model_name)

    # Translate the test data
    hypotheses = [translate_text(model, tokenizer, sentence) for sentence in sources]

    # Compute evaluation metrics
    bleu_score = compute_bleu(references, hypotheses)
    comet_score = compute_comet(references, hypotheses, sources)

    # Debugging output for translations
    print("\n--- Translation Debugging Output ---")
    for src, hyp, ref in zip(sources, hypotheses, references):
        print(f"🔹 Source: {src}")
        print(f"🔹 Hypothesis: {hyp}")
        print(f"🔹 Reference: {ref}")
        print("----")

    # Save results
    result = {
        "model": model_name,
        "BLEU": round(bleu_score, 2),
        "COMET": round(comet_score, 2),
    }

    print(result)

    # Append results to JSON file
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "r") as f:
            try:
                results = json.load(f)
            except json.JSONDecodeError:
                results = []
    else:
        results = []

    results.append(result)

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=4)

    # Assertions to ensure translation quality is reasonable
    assert bleu_score > 10, f"BLEU score too low for {model_name}: {bleu_score}"
    assert comet_score > 0.5, f"COMET score too low for {model_name}: {comet_score}"
