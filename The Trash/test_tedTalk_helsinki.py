import os
import csv
import pytest
import re
from datasets_loader.load_ted import load_ted_data  # Import the function from load_ted.py
from helpers.model_loader import load_model, translate_text
from helpers.evaluation import compute_bleu, compute_comet  # Ensure helpers are correct

# Define the path for results inside the 'data' folder
RESULTS_CSV = "data/ted_talks_test_results.csv"
os.makedirs("data", exist_ok=True)

# Dictionary of models to test
MODELS_TO_TEST = {
    "English-Albanian": "Helsinki-NLP/opus-mt-en-sq",
    "English-Bulgarian": "Helsinki-NLP/opus-mt-en-bg",
    "English-Czech": "Helsinki-NLP/opus-mt-en-cs",
    "English-Danish": "Helsinki-NLP/opus-mt-en-da",
    "English-Dutch": "Helsinki-NLP/opus-mt-en-nl",
    "English-Estonian": "Helsinki-NLP/opus-mt-en-et",
    "English-Finnish": "Helsinki-NLP/opus-mt-en-fi",
    "English-French": "Helsinki-NLP/opus-mt-en-fr",
    "English-German": "Helsinki-NLP/opus-mt-en-de",
    "English-Greek": "Helsinki-NLP/opus-mt-en-el",
    "English-Croatian": "Helsinki-NLP/opus-mt-en-hr",
    "English-Hungarian": "Helsinki-NLP/opus-mt-en-hu",
    "English-Italian": "Helsinki-NLP/opus-mt-en-it",
    "English-Icelandic": "Helsinki-NLP/opus-mt-en-is",
    "English-Latvian": "Helsinki-NLP/opus-mt-tc-big-en-lv",
    "English-Lithuanian": "Helsinki-NLP/opus-mt-tc-big-en-lt",
    "English-Macedonian": "Helsinki-NLP/opus-mt-en-mk",
    "English-Polish": "Helsinki-NLP/opus-mt-en-pl",
    "English-Portuguese": "Helsinki-NLP/opus-mt-tc-big-en-pt",
    "English-Romanian": "Helsinki-NLP/opus-mt-en-ro",
    "English-Slovak": "Helsinki-NLP/opus-mt-en-sk",
    "English-Slovenian": "Helsinki-NLP/opus-mt-en-sl",
    "English-Spanish": "Helsinki-NLP/opus-mt-en-es",
    "English-Swedish": "Helsinki-NLP/opus-mt-en-sv",
    "English-Turkish": "Helsinki-NLP/opus-mt-tc-big-en-tr",
}

def extract_lang_pair(model_name):
    """
    Extracts source and target language pairs from a given model name.
    """
    match = re.search(r"opus-mt(?:-[a-z]+)*-(\w+)-(\w+)", model_name)
    if match:
        source_lang, target_lang = match.groups()
        return source_lang, target_lang
    else:
        raise ValueError(f"Could not extract language pair from model: {model_name}")

def write_to_csv(model_name, target_lang, bleu_score, comet_score):
    """Append translation results to CSV file."""
    file_exists = os.path.isfile(RESULTS_CSV)

    with open(RESULTS_CSV, mode='a', newline='', encoding="utf-8") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Model", "Target Language", "BLEU Score", "COMET Score"])  # Write header

        writer.writerow([
            model_name,
            target_lang,
            "NA" if bleu_score is None else round(bleu_score, 2),
            "NA" if comet_score is None else round(comet_score, 2),
        ])

@pytest.mark.parametrize("lang_pair,model_name", MODELS_TO_TEST.items())
def test_translation_quality(lang_pair, model_name):
    """Test translation models using TED Talks data and log results in CSV."""
    
    print(f"🔹 Testing model: {model_name} ({lang_pair})")  # Debugging output

    # Load model & tokenizer
    model, tokenizer = load_model(model_name)

    # Extract language pair
    _, target_lang = extract_lang_pair(model_name)

    # Load TED dataset
    sources, references = load_ted_data(target_lang)

    if not sources or not references:
        print(f"⚠️ Skipping {model_name} ({target_lang}): No dataset found. Logging 'NA' to CSV.")
        write_to_csv(model_name, target_lang, None, None)  # Save "NA" result
        pytest.skip(f"Skipping {model_name} ({target_lang}): No test data available.")
        return

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

    # Save results to CSV
    write_to_csv(model_name, target_lang, bleu_score, comet_score)

    assert bleu_score > 10, f"BLEU score too low for {model_name}: {bleu_score}"
    assert comet_score > 0.5, f"COMET score too low for {model_name}: {comet_score}"
