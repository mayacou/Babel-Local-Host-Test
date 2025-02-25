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
    "English-Croatian": "Helsinki-NLP/opus-mt-tc-base-en-sh",
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
    "English-Latvian": "Helsinki-NLP/opus-mt-tc-big-en-lv",
    "English-Lithuanian": "Helsinki-NLP/opus-mt-tc-big-en-lt",
    "English-Macedonian": "Helsinki-NLP/opus-mt-en-mk",
    "English-Polish": "Helsinki-NLP/opus-mt-en-sla",
    "English-Portuguese": "Helsinki-NLP/opus-mt-tc-big-en-pt",
    "English-Romanian": "Helsinki-NLP/opus-mt-en-ro",
    "English-Slovak": "Helsinki-NLP/opus-mt-en-sk",
    "English-Slovenian": "Helsinki-NLP/opus-mt-en-sla",
    "English-Spanish": "Helsinki-NLP/opus-mt-en-es",
    "English-Swedish": "Helsinki-NLP/opus-mt-en-sv",
    "English-Turkish": "Helsinki-NLP/opus-mt-tc-big-en-tr",
}

def extract_lang_pair(model_name):
    """
    Extracts the source and target language codes from the model name.
    Example: "Helsinki-NLP/opus-mt-en-fr" -> ("en", "fr")
    Example: "Helsinki-NLP/opus-mt-tc-big-en-de" -> ("en", "de")
    """
    match = re.search(r"opus-mt(?:-tc-big)?-(\w+)-(\w+)", model_name)
    if match:
        src_lang, tgt_lang = match.groups()
        return src_lang, tgt_lang
    raise ValueError(f"Could not extract language pair from model: {model_name}")

@pytest.mark.parametrize("lang_pair,model_name", MODELS_TO_TEST.items())
def test_translation_quality(lang_pair, model_name):
    """Test translation models on TED Talks dataset."""
    
    # Extract source and target languages
    src_lang, tgt_lang = extract_lang_pair(model_name)

    # TED Talks dataset requires specifying a language pair and year
    dataset_config = {"language_pair": (src_lang, tgt_lang), "year": "2014"}

    # Load TED Talks dataset dynamically
    try:
        dataset = load_dataset("IWSLT/ted_talks_iwslt", **dataset_config)
    except ValueError as e:
        pytest.skip(f"Skipping {lang_pair} due to missing dataset config: {e}")

    # Load a small subset for testing
    test_samples = dataset["train"].select(range(5))  # Use 100 samples

    # Load translation model and tokenizer
    model = MarianMTModel.from_pretrained(model_name)
    tokenizer = MarianTokenizer.from_pretrained(model_name)

    references = []
    translations = []

    for sample in test_samples:
        src_text = sample['translation'][src_lang]
        tgt_text = sample['translation'][tgt_lang]

        # Tokenize input
        inputs = tokenizer(src_text, return_tensors="pt", padding=True, truncation=True)

        # Generate translation
        translated_tokens = model.generate(**inputs)
        translated_text = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]

        references.append([tgt_text])  # BLEU expects list of lists
        translations.append(translated_text)

    # Compute BLEU score
    bleu_score = corpus_bleu(translations, references).score
    print(f"BLEU Score for {lang_pair}: {bleu_score}")

    # Ensure BLEU score is reasonable
    assert bleu_score > 10, f"Low BLEU score for {lang_pair}: {bleu_score}"

    # Save results to JSON
    results = {
        "lang_pair": lang_pair,
        "model": model_name,
        "bleu_score": bleu_score,
    }

    with open(RESULTS_FILE, "a") as f:
        json.dump(results, f)
        f.write("\n")
