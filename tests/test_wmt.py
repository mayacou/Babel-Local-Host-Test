import json
import os
import pytest
import re
from datasets import load_dataset
from helpers.model_loader import load_model, translate_text
from helpers.evaluation import compute_bleu, compute_comet

# Define the path for the results file inside the data folder
RESULTS_FILE = "data/wmt_test_results.json"

# Ensure the `data/` directory exists
os.makedirs("data", exist_ok=True)

MODELS_TO_TEST = {
    # Add all models here...
    "Helsinki-NLP/opus-mt-en-sq",  # English → Albanian
    "Helsinki-NLP/opus-mt-en-bg",  # English → Bulgarian
    "Helsinki-NLP/opus-mt-en-hr",  # English → Croatian
    "Helsinki-NLP/opus-mt-en-cs",  # English → Czech
    "Helsinki-NLP/opus-mt-en-da",  # English → Danish
    "Helsinki-NLP/opus-mt-en-nl",  # English → Dutch
    "Helsinki-NLP/opus-mt-en-et",  # English → Estonian
    "Helsinki-NLP/opus-mt-en-fi",  # English → Finnish
    "Helsinki-NLP/opus-mt-en-fr",  # English → French
    "Helsinki-NLP/opus-mt-en-de",  # English → German
    "Helsinki-NLP/opus-mt-en-el",  # English → Greek
    "Helsinki-NLP/opus-mt-en-hu",  # English → Hungarian
    "Helsinki-NLP/opus-mt-en-is",  # English → Icelandic
    "Helsinki-NLP/opus-mt-en-it",  # English → Italian
    "Helsinki-NLP/opus-mt-en-lv",  # English → Latvian
    "Helsinki-NLP/opus-mt-en-lt",  # English → Lithuanian
    "Helsinki-NLP/opus-mt-en-mk",  # English → Macedonian
    "Helsinki-NLP/opus-mt-en-no",  # English → Norwegian
    "Helsinki-NLP/opus-mt-en-pl",  # English → Polish
    "Helsinki-NLP/opus-mt-en-pt",  # English → Portuguese
    "Helsinki-NLP/opus-mt-en-ro",  # English → Romanian
    "Helsinki-NLP/opus-mt-en-sk",  # English → Slovak
    "Helsinki-NLP/opus-mt-en-sl",  # English → Slovenian
    "Helsinki-NLP/opus-mt-en-es",  # English → Spanish
    "Helsinki-NLP/opus-mt-en-sv",  # English → Swedish
    "Helsinki-NLP/opus-mt-en-tr",  # English → Turkish
    #24 of 28 languages found for wmt24++ (this dataset) missing English(bc its source lang), Luxembourgish, Montenegrin, norwegian
}

WMT_DATASET = "wmt24++"

def load_wmt_data(model_name):
    """Load test data dynamically based on the model's language pair."""
    print(f"Loading dataset for model: {model_name}")

    # Auto-extract source & target languages from model name
    match = re.search(r"opus-mt-(\w+)-(\w+)", model_name)
    if not match:
        raise ValueError(f"Could not extract language pair from model: {model_name}")

    source_lang, target_lang = match.groups()
    dataset_name = f"{target_lang}-{source_lang}"  # Swap order for Hugging Face format

    print(f"Detected Language Pair: {source_lang} → {target_lang} | Using Dataset: {dataset_name}")

    # Load dataset
    dataset = load_dataset("wmt14", dataset_name)

    # Extract translations
    test_samples = dataset["test"]["translation"][:5]
    print("First Sample:", test_samples[0])  # Debugging print

    sources = [sample[source_lang] for sample in test_samples]
    references = [sample[target_lang] for sample in test_samples]

    return sources, references

@pytest.mark.parametrize("model_name", MODELS_TO_TEST)
def test_translation_quality(model_name, request):
    """Test translation models using WMT data and log results in JSON."""
    model, tokenizer = load_model(model_name)
    sources, references = load_wmt_data(model_name)  # Pass model name as argument

    hypotheses = [translate_text(model, tokenizer, sentence) for sentence in sources]

    # Compute evaluation metrics
    bleu_score = compute_bleu(references, hypotheses)
    comet_score = compute_comet(references, hypotheses, sources)

    # Debugging print statements
    print("\n--- Translation Debugging Output ---")
    for src, hyp, ref in zip(sources, hypotheses, references):
        print(f"🔹 Source: {src}")
        print(f"🔹 Hypothesis: {hyp}")
        print(f"🔹 Reference: {ref}")
        print("----")

    result = {
        "model": model_name,
        "BLEU": round(bleu_score, 2),
        "COMET": round(comet_score, 2),
    }

    print(result)

    # Save results
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

    assert bleu_score > 25, "BLEU score is too low!"
    assert comet_score > 0.5, "COMET score is too low!"
