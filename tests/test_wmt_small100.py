import os
import csv
import pytest
import random
from datasets_loader.load_wmt import load_wmt_data  # Import the function from load_wmt.py
from transformers import M2M100ForConditionalGeneration
from helpers.tokenization_small100 import SMALL100Tokenizer
from helpers.evaluation import compute_bleu, compute_comet  # Ensure helpers are correct

# Define the path for the results CSV file
RESULTS_CSV = "data/small100_test_results.csv"
TRANSLATIONS_CSV = "translation_results/small100_translations.csv"

# Ensure the data/ directory exists
os.makedirs("data", exist_ok=True)

# List of target languages to test
LANGUAGES_TO_TEST = {
    "sq": "sq_AL", "bg": "bg_BG", "hr": "hr_HR", "cs": "cs_CZ", "da": "da_DK",
    "nl": "nl_NL", "et": "et_EE", "fi": "fi_FI", "fr": "fr_FR", "de": "de_DE",
    "el": "el_GR", "hu": "hu_HU", "is": "is_IS", "it": "it_IT", "lv": "lv_LV",
    "lt": "lt_LT", "lb": "lb_LU", "mk": "mk_MK", "no": "no_NO", "pl": "pl_PL",
    "pt": "pt_PT", "ro": "ro_RO", "sk": "sk_SK", "sl": "sl_SI", "es": "es_MX",
    "sv": "sv_SE", "tr": "tr_TR"
}

def write_to_csv(model, language, bleu, comet):
    """Append a row to the CSV file."""
    file_exists = os.path.isfile(RESULTS_CSV)
    with open(RESULTS_CSV, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Model", "Language", "Bleu", "Comet"])
        writer.writerow([model, language, bleu, comet])

def write_translations_to_csv(model, language, sources, hypotheses, references):
    """Append source sentences, translations, and references to a separate CSV file."""
    file_exists = os.path.isfile(TRANSLATIONS_CSV)
    with open(TRANSLATIONS_CSV, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Model", "Language", "Source Sentence", "Translation", "Reference Sentence"])
        for source, hypothesis, reference in zip(sources, hypotheses, references):
            writer.writerow([model, language, source, hypothesis, reference])

@pytest.mark.parametrize("target_lang_code", LANGUAGES_TO_TEST.keys())
def test_translation_quality(target_lang_code, request):
    """Test Small100 translations using WMT data and log results in CSV."""
    print(f"🛠️ Loading Small100 model for {target_lang_code}...")

    # Load Small100 model
    model = M2M100ForConditionalGeneration.from_pretrained("alirezamsh/small100")
    tokenizer = SMALL100Tokenizer.from_pretrained("alirezamsh/small100", tgt_lang=target_lang_code)

    sources, references = load_wmt_data(target_lang_code)  # Now using load_wmt.py

    if not sources or not references:
        print(f"⚠️ Skipping {target_lang_code}: No dataset found. Logging 'NA' to CSV.")
        write_to_csv("alirezamsh/small100", target_lang_code, "NA", "NA")
        pytest.skip(f"Skipping {target_lang_code}: No test data available.")
        return

    hypotheses = []
    for sentence in sources:
        model_inputs = tokenizer(sentence, return_tensors="pt")
        output_tokens = model.generate(**model_inputs, num_beams=5, max_length=256)
        hypothesis = tokenizer.batch_decode(output_tokens, skip_special_tokens=True)[0]
        hypotheses.append(hypothesis)

    # Compute evaluation metrics
    bleu_score = compute_bleu(references, hypotheses)
    comet_score = compute_comet(references, hypotheses, sources)

    print(result := {
        "model": "alirezamsh/small100",
        "target_language": target_lang_code,
        "BLEU": round(bleu_score, 2),
        "COMET": round(comet_score, 2),
    })

    write_to_csv("alirezamsh/small100", target_lang_code, round(bleu_score, 2), round(comet_score, 2))
    write_translations_to_csv("alirezamsh/small100", target_lang_code, sources, hypotheses, references)

    assert bleu_score > 10, "BLEU score is too low!"
    assert comet_score > 0.5, "COMET score is too low!"

