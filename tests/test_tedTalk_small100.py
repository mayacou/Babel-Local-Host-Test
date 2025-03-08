import os
import csv
import pytest
import random
from datasets_loader.load_tedTalk import load_tedTalk_data  # Import the updated function
from transformers import M2M100ForConditionalGeneration
from helpers.tokenization_small100 import SMALL100Tokenizer
from helpers.evaluation import compute_bleu, compute_comet  # Ensure helpers are correct

# Define the path for the results CSV file
RESULTS_CSV = "data/small100_tedTalk_results.csv"

# Ensure the data/ directory exists
os.makedirs("data", exist_ok=True)

# ✅ Supported TED Talks languages (format must match dataset)
LANGUAGES_TO_TEST = [
    "sq", "bg", "hr", "cs", "da", "nl", "et", "fi", "fr", "de", "el",
    "hu", "is", "it", "lv", "lt", "mk", "pl", "pt", "ro", "sk", "sl",
    "es", "sv", "tr"
]

def write_to_csv(model, language, bleu, comet):
    """Append a row to the CSV file."""
    file_exists = os.path.isfile(RESULTS_CSV)

    with open(RESULTS_CSV, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Model", "Language", "Bleu", "Comet"])
        writer.writerow([model, language, bleu, comet])

@pytest.mark.parametrize("target_lang_code", LANGUAGES_TO_TEST)
def test_translation_quality(target_lang_code, request):
    """✅ Test Small100 translations using TED Talks dataset."""
    print(f"🛠️ Loading Small100 model for {target_lang_code}...")

    # ✅ Load Small100 model
    model = M2M100ForConditionalGeneration.from_pretrained("alirezamsh/small100")
    tokenizer = SMALL100Tokenizer.from_pretrained("alirezamsh/small100", tgt_lang=target_lang_code)

    # ✅ Load TED dataset using `en` as the source language
    sources, references = load_tedTalk_data(target_lang_code, source_lang="en")

    if not sources or not references:
        print(f"⚠️ Skipping {target_lang_code}: No dataset found. Logging 'NA' to CSV.")
        write_to_csv("alirezamsh/small100", target_lang_code, "NA", "NA")  # ✅ Write before skipping
        pytest.skip(f"Skipping {target_lang_code}: No test data available.")  # ✅ Skip afterwards

    hypotheses = []
    for sentence in sources:
        model_inputs = tokenizer(sentence, return_tensors="pt")
        output_tokens = model.generate(**model_inputs, num_beams=5, max_length=256)
        hypothesis = tokenizer.batch_decode(output_tokens, skip_special_tokens=True)[0]
        hypotheses.append(hypothesis)

    # ✅ FIX: Handle edge cases where sources might be empty
    if not hypotheses:
        print(f"⚠️ No translations generated for {target_lang_code}. Writing 'NA' results.")
        write_to_csv("alirezamsh/small100", target_lang_code, "NA", "NA")
        return

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

    assert bleu_score > 10, "BLEU score is too low!"
    assert comet_score > 0.5, "COMET score is too low!"
