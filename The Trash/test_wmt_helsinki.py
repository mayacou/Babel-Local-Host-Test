import os
import pytest
import re
import csv
from helpers.model_loader import load_model, translate_text
from helpers.evaluation import compute_bleu, compute_comet
from datasets_loader.load_wmt import load_wmt_data  # Importing the function from load_wmt.py

# Define the path for the results file inside the data folder
RESULTS_FILE = "data/wmt_test_results.csv"

# Ensure the data/ directory exists
os.makedirs("data", exist_ok=True)

MODELS_TO_TEST = {
    "Helsinki-NLP/opus-mt-en-sq",
    "Helsinki-NLP/opus-mt-en-bg",
    "Helsinki-NLP/opus-mt-tc-base-en-sh",
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
    "Helsinki-NLP/opus-mt-en-sla",
    "Helsinki-NLP/opus-mt-tc-big-en-pt",
    "Helsinki-NLP/opus-mt-en-ro",
    "Helsinki-NLP/opus-mt-en-sk",
    "Helsinki-NLP/opus-mt-en-es",
    "Helsinki-NLP/opus-mt-en-sv",
    "Helsinki-NLP/opus-mt-tc-big-en-tr",
}

def extract_language_pair_from_model(model_name):
    """
    Extracts the target language from the model name using regex.
    Returns a list of target languages if "sla" is detected.
    """
    match = re.search(r"opus-mt(?:-[a-z]+)*-en-([\w]+)", model_name)
    if not match:
        raise ValueError(f"Could not extract language pair from model: {model_name}")

    target_lang = match.group(1)  # Extracts the target language code
    
    # If "sla" is detected, return multiple languages
    if target_lang == "sla":
        return ["hr", "pl", "sl"]  # Croatian, Polish, Slovenian

    return [target_lang]  # Wrap in a list for consistency

def save_results_to_csv(model_name, target_lang, bleu_score, comet_score):
    """
    Save model results to wmt_test_results.csv.
    If BLEU or COMET is 'NA', it means the dataset was not found.
    """
    file_exists = os.path.isfile(RESULTS_FILE)

    with open(RESULTS_FILE, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        # Write header only if the file is new
        if not file_exists:
            writer.writerow(["Model", "Target Language", "BLEU Score", "COMET Score"])

        writer.writerow([
            model_name,
            target_lang,
            "NA" if bleu_score is None else round(bleu_score, 2),
            "NA" if comet_score is None else round(comet_score, 2),
        ])

@pytest.mark.parametrize("model_name", MODELS_TO_TEST)
def test_translation_quality(model_name, request):
    """Test translation models using WMT data and log results in CSV."""
    model, tokenizer = load_model(model_name)
    
    # Extract language pair(s)
    target_langs = extract_language_pair_from_model(model_name)

    for target_lang in target_langs:
        print(f"🔍 Extracted language pair for {model_name}: {target_lang}")  # Debugging
        sources, references = load_wmt_data(target_lang)  # Now using the function from load_wmt.py

        if not sources or not references:
            print(f"⚠️ Skipping {model_name} ({target_lang}): No dataset found. Logging 'NA' to CSV.")
            save_results_to_csv(model_name, target_lang, None, None)  # Save "NA" result
            pytest.skip(f"Skipping {model_name} ({target_lang}): No test data available.")
            continue

        # Generate translations with the target language token
        hypotheses = [translate_text(model, tokenizer, sentence, target_lang) for sentence in sources]

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

        # Save results to CSV
        save_results_to_csv(model_name, target_lang, bleu_score, comet_score)

        assert bleu_score > 10, "BLEU score is too low!"
        assert comet_score > 0.5, "COMET score is too low!"
