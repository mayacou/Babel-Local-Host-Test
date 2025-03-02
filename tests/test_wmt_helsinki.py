import json
import os
import pytest
import re
from datasets import get_dataset_config_names, load_dataset
from helpers.model_loader import load_model, translate_text
from helpers.evaluation import compute_bleu, compute_comet

# Define the path for the results file inside the data folder
RESULTS_FILE = "data/wmt_test_results.json"

# Ensure the data/ directory exists
os.makedirs("data", exist_ok=True)

MODELS_TO_TEST = {
    # Add all models here...
    "Helsinki-NLP/opus-mt-en-sq",
    "Helsinki-NLP/opus-mt-en-bg",
    #"Helsinki-NLP/opus-mt-tc-base-en-sh",
    "Helsinki-NLP/opus-mt-en-cs",
    "Helsinki-NLP/opus-mt-en-da",
    "Helsinki-NLP/opus-mt-en-nl",
    "Helsinki-NLP/opus-mt-en-et",
    "Helsinki-NLP/opus-mt-en-fi",
    "Helsinki-NLP/opus-mt-en-fr",
    "Helsinki-NLP/opus-mt-en-de",
    "Helsinki-NLP/opus-mt-en-el",
    "Helsinki-NLP/opus-mt-en-hu",
    "Helsinki-NLP/opus-mt-en-it",
    "Helsinki-NLP/opus-mt-en-is",
    "Helsinki-NLP/opus-mt-tc-big-en-lv",
    "Helsinki-NLP/opus-mt-tc-big-en-lt",
    "Helsinki-NLP/opus-mt-en-mk",
    #"Helsinki-NLP/opus-mt-en-sla",
    "Helsinki-NLP/opus-mt-tc-big-en-pt",
    "Helsinki-NLP/opus-mt-en-ro",
    "Helsinki-NLP/opus-mt-en-sk",
    #"Helsinki-NLP/opus-mt-en-sla",
    "Helsinki-NLP/opus-mt-en-es",
    "Helsinki-NLP/opus-mt-en-sv",
    "Helsinki-NLP/opus-mt-tc-big-en-tr",
}

WMT_DATASET = "google/wmt24pp"

def load_wmt_data(model_name):
    """Load test data dynamically based on the model's language pair."""
    print(f"Loading dataset for model: {model_name}")

    # Auto-extract source & target languages from model name
    match = re.search(r"opus-mt(?:-[a-z]+)*-(\w+)-(\w+)", model_name)
    if not match:
        raise ValueError(f"Could not extract language pair from model: {model_name}")

    source_lang, target_lang = match.groups()

    # Ensure correct dataset naming using the full region codes
    LANGUAGE_CODE_MAP = {
        "ar": "ar_SA", "bg": "bg_BG", "bn": "bn_IN", "ca": "ca_ES", "cs": "cs_CZ",
        "da": "da_DK", "de": "de_DE", "el": "el_GR", "es": "es_MX", "et": "et_EE",
        "fi": "fi_FI", "fr": "fr_FR", "hi": "hi_IN", "hu": "hu_HU", "is": "is_IS",
        "it": "it_IT", "lt": "lt_LT", "lv": "lv_LV", "nl": "nl_NL", "pl": "pl_PL",
        "pt": "pt_PT", "ro": "ro_RO", "ru": "ru_RU", "sk": "sk_SK", "sv": "sv_SE",
        "tr": "tr_TR", "uk": "uk_UA", "zh": "zh_CN",
    }

    if target_lang not in LANGUAGE_CODE_MAP:
        print(f"⚠️ Skipping {model_name}: Language {target_lang} not found in mapping.")
        pytest.skip(f"Skipping {model_name}: Language {target_lang} not found.")

    dataset_name = f"en-{LANGUAGE_CODE_MAP[target_lang]}"  # Correct naming format

    print(f"🔎 Detected Language Pair: en → {target_lang} | Using Dataset: {dataset_name}")

    # Load dataset
    try:
        dataset = load_dataset(WMT_DATASET, dataset_name)
    except ValueError:
        print(f"⚠️ Skipping {model_name}: No dataset found for {dataset_name}.")
        pytest.skip(f"Skipping {model_name}: No dataset found.")

    # Use "train" split since others are empty
    if "train" in dataset and len(dataset["train"]) > 0:
        split_name = "train"
    else:
        print(f"⚠️ Skipping {model_name}: No usable split found.")
        pytest.skip(f"Skipping {model_name}: No usable split found.")

    # Convert dataset to list format
    test_samples = list(dataset[split_name])[:5]  # Convert to list and take 5 samples

    if not test_samples:
        print(f"⚠️ Skipping {model_name}: No test samples found.")
        pytest.skip(f"Skipping {model_name}: No test samples found.")

    # Debugging print
    print(f"📝 First Sample: {test_samples[0]}")  

    sources = [sample["source"] for sample in test_samples]
    references = [sample["target"] for sample in test_samples]  # Use post-edit as reference

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

    assert bleu_score > 10, "BLEU score is too low!"
    assert comet_score > 0.5, "COMET score is too low!"
