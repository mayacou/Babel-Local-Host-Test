import os
import csv
import pytest
import torch
from datasets_loader.load_europarl import load_europarl_data
from transformers import M2M100ForConditionalGeneration
from helpers.tokenization_small100 import SMALL100Tokenizer
from helpers.evaluation import compute_bleu, compute_comet

# Define paths for results
RESULTS_CSV = "data/europarl_test_results.csv"
TRANSLATIONS_CSV = "translation_results/europarl_small100_translations.csv"

# Ensure directories exist
os.makedirs("data", exist_ok=True)
os.makedirs("translation_results", exist_ok=True)

# List of target languages to test (Europarl-supported languages)
EUROPARL_LANGUAGES_TO_TEST = [
    "bg", "cs", "da", "nl", "et", "fi", "fr", "de", "el", "hu",
    "it", "lv", "lt", "pl", "pt", "ro", "sk", "sl", "es", "sv", "tr"
]

# ✅ Use CUDA if available, otherwise default to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    print(f"🚀 Using NVIDIA GPU: {torch.cuda.get_device_name(0)}")
else:
    print("⚠️ No NVIDIA GPU detected, using CPU instead.")

def write_to_csv(model, language, bleu, comet):
    """Append a row to the results CSV file."""
    with open(RESULTS_CSV, mode='a', newline='') as file:
        writer = csv.writer(file)
        if file.tell() == 0:  # Write header only if the file is empty
            writer.writerow(["Model", "Language", "BLEU", "COMET"])
        writer.writerow([model, language, bleu, comet])

def write_translations_to_csv(model, language, sources, hypotheses, references):
    """Append source sentences, translations, and references to a separate CSV file."""
    with open(TRANSLATIONS_CSV, mode='a', newline='') as file:
        writer = csv.writer(file)
        if file.tell() == 0:  # Write header only if the file is empty
            writer.writerow(["Model", "Language", "Source Sentence", "Translation", "Reference Sentence"])
        for source, hypothesis, reference in zip(sources, hypotheses, references):
            writer.writerow([model, language, source, hypothesis, reference])

@pytest.mark.parametrize("target_lang_code", EUROPARL_LANGUAGES_TO_TEST)
def test_translation_quality(target_lang_code):
    """Test Small100 translations using Europarl data (en → target) and log results in CSV."""
    print(f"🛠️ Loading Small100 model for {target_lang_code}...")

    # ✅ Load model and tokenizer on GPU (if available)
    model = M2M100ForConditionalGeneration.from_pretrained("alirezamsh/small100").to(device)
    tokenizer = SMALL100Tokenizer.from_pretrained("alirezamsh/small100", tgt_lang=target_lang_code)

    # ✅ Convert model to FP16 (half precision) for faster inference (optional)
    if device.type == "cuda":
        model = model.half()

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
        # ✅ Move tokenized inputs to GPU for faster processing
        model_inputs = tokenizer(sentence.strip(), return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)

        # ✅ Disable gradient computation for inference (saves memory & speeds up inference)
        with torch.no_grad():
            output_tokens = model.generate(
                **model_inputs, 
                num_beams=10, 
                max_length=256, 
                early_stopping=True
            )

        # ✅ Decode output and move to CPU to avoid unnecessary GPU usage
        hypothesis = tokenizer.batch_decode(output_tokens, skip_special_tokens=True)[0].strip().lower()
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

    # ✅ Write results and translations to CSVs
    write_to_csv("alirezamsh/small100", target_lang_code, round(bleu_score, 2), round(comet_score, 2))
    write_translations_to_csv("alirezamsh/small100", target_lang_code, sources, hypotheses, references)

    assert bleu_score > 5, f"BLEU score is too low! ({bleu_score})"
    assert comet_score > 0.5, f"COMET score is too low! ({comet_score})"
