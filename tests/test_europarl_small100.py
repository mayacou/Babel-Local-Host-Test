import os
import csv
import pytest
import random
from datasets_loader.load_europarl import load_europarl_data  # Import the function from load_europarl.py
from transformers import M2M100ForConditionalGeneration
from helpers.tokenization_small100 import SMALL100Tokenizer
from helpers.evaluation import compute_bleu, compute_comet  # Ensure helpers are correct

# Define the path for the results CSV file
RESULTS_CSV = "data/europarl_test_results.csv"

# Ensure the data/ directory exists
os.makedirs("data", exist_ok=True)

# List of target languages to test (Europarl-supported languages)
EUROPARL_LANGUAGES_TO_TEST = [
    "bg", "cs", "da", "nl", "et", "fi", "fr", "de", "el", "hu",
    "it", "lv", "lt", "pl", "pt", "ro", "sk", "sl", "es", "sv", "tr"
]

def write_to_csv(model, language, bleu, comet):
    """Append a row to the CSV file."""
    file_exists = os.path.isfile(RESULTS_CSV)

    with open(RESULTS_CSV, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Model", "Language", "BLEU", "COMET"])  # Write header if file does not exist
        writer.writerow([model, language, bleu, comet])

@pytest.mark.parametrize("target_lang_code", EUROPARL_LANGUAGES_TO_TEST)
def test_translation_quality(target_lang_code):
    """Test Small100 translations using Europarl data (en → target) and log results in CSV."""
    print(f"🛠️ Loading Small100 model for {target_lang_code}...")

    # ✅ Use Small100 Model & Tokenizer
    model = M2M100ForConditionalGeneration.from_pretrained("alirezamsh/small100")
    tokenizer = SMALL100Tokenizer.from_pretrained("alirezamsh/small100", tgt_lang=target_lang_code)

    # Load test data (ensuring en → target_lang_code)
    sources, references = load_europarl_data(target_lang_code)

    if not sources or not references:
        print(f"⚠️ Skipping {target_lang_code}: No dataset found. Logging 'NA' to CSV.")
        write_to_csv("alirezamsh/small100", target_lang_code, "NA", "NA")
        pytest.skip(f"Skipping {target_lang_code}: No test data available.")
        return

    # Normalize references
    references = [ref.strip().lower() for ref in references]

    hypotheses = []
    for sentence in sources:
        model_inputs = tokenizer(sentence.strip(), return_tensors="pt", padding=True, truncation=True, max_length=256)
        output_tokens = model.generate(**model_inputs, num_beams=10, max_length=256, early_stopping=True)
        hypothesis = tokenizer.batch_decode(output_tokens, skip_special_tokens=True)[0].strip().lower()
        hypotheses.append(hypothesis)

    bleu_score = compute_bleu(references, hypotheses)
    comet_score = compute_comet(references, hypotheses, sources)

    print(result := {
        "model": "alirezamsh/small100",
        "target_language": target_lang_code,
        "BLEU": round(bleu_score, 2),
        "COMET": round(comet_score, 2),
    })

    write_to_csv("alirezamsh/small100", target_lang_code, round(bleu_score, 2), round(comet_score, 2))

    assert bleu_score > 5, f"BLEU score is too low! ({bleu_score})"
    assert comet_score > 0.5, f"COMET score is too low! ({comet_score})"

