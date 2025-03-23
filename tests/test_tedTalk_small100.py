import os
import csv
import pytest
import torch
from datasets_loader.load_tedTalk import load_tedTalk_data  # Import the updated function
from transformers import M2M100ForConditionalGeneration
from helpers.tokenization_small100 import SMALL100Tokenizer
from helpers.evaluation import compute_bleu, compute_comet  # Ensure helpers are correct

# Define the path for the results CSV file
RESULTS_CSV = "data/small100_tedTalk_results.csv"
TRANSLATIONS_CSV = "translation_results/small100_tedTalk_translations.csv"

# Ensure the data/ directory exists
os.makedirs("translation_results", exist_ok=True)

# ✅ Supported TED Talks languages (format must match dataset)
LANGUAGES_TO_TEST = [
    "sq", "bg", "hr", "cs", "da", "nl", "et", "fi", "fr", "de", "el",
    "hu", "is", "it", "lv", "lt", "mk", "pl", "pt", "ro", "sk", "sl",
    "es", "sv", "tr"
]

# ✅ Use CUDA if available, otherwise default to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    print(f"🚀 Using NVIDIA GPU: {torch.cuda.get_device_name(0)}")
else:
    print("⚠️ No NVIDIA GPU detected, using CPU instead.")

@pytest.mark.parametrize("target_lang_code", LANGUAGES_TO_TEST)
def test_translation_quality(target_lang_code, request):
    """✅ Test Small100 translations using TED Talks dataset."""
    print(f"🛠️ Loading Small100 model for {target_lang_code}...")

    # ✅ Load Small100 model and move to GPU
    model = M2M100ForConditionalGeneration.from_pretrained("alirezamsh/small100").to(device)
    tokenizer = SMALL100Tokenizer.from_pretrained("alirezamsh/small100", tgt_lang=target_lang_code)

    # ✅ Load TED dataset using `en` as the source language
    sources, references = load_tedTalk_data(target_lang_code, source_lang="en")

    if not sources or not references:
        print(f"⚠️ Skipping {target_lang_code}: No dataset found. Logging 'NA' to CSV.")
        
        # ✅ Open results CSV once for the entire test
        with open(RESULTS_CSV, mode="a", newline="") as file:
            writer = csv.writer(file)
            if os.stat(RESULTS_CSV).st_size == 0:
                writer.writerow(["Model", "Language", "BLEU", "COMET"])
            writer.writerow(["alirezamsh/small100", target_lang_code, "NA", "NA"])
        
        pytest.skip(f"Skipping {target_lang_code}: No test data available.")

    hypotheses = []
    for sentence in sources:
        model_inputs = tokenizer(sentence, return_tensors="pt").to(device)  # ✅ Move input tensors to GPU
        output_tokens = model.generate(**model_inputs, num_beams=10, max_length=256, early_stopping=True)
        hypothesis = tokenizer.batch_decode(output_tokens, skip_special_tokens=True)[0]
        hypotheses.append(hypothesis)

    # ✅ FIX: Handle edge cases where sources might be empty
    if not hypotheses:
        print(f"⚠️ No translations generated for {target_lang_code}. Writing 'NA' results.")
        with open(RESULTS_CSV, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["alirezamsh/small100", target_lang_code, "NA", "NA"])
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

    # ✅ Open CSV files **once** for the entire test loop
    with open(RESULTS_CSV, mode="a", newline="") as results_file, open(TRANSLATIONS_CSV, mode="a", newline="") as trans_file:
        results_writer = csv.writer(results_file)
        trans_writer = csv.writer(trans_file)

        # ✅ Write headers if files are empty
        if os.stat(RESULTS_CSV).st_size == 0:
            results_writer.writerow(["Model", "Language", "BLEU", "COMET"])

        if os.stat(TRANSLATIONS_CSV).st_size == 0:
            trans_writer.writerow(["Model", "Language", "Source Sentence", "Translation", "Reference Sentence"])

        # ✅ Save BLEU/COMET results
        results_writer.writerow(["alirezamsh/small100", target_lang_code, round(bleu_score, 2), round(comet_score, 2)])

        # ✅ Save translations
        for src, hyp, ref in zip(sources, hypotheses, references):
            trans_writer.writerow(["alirezamsh/small100", target_lang_code, src, hyp, ref])

        # ✅ Ensure data is flushed to file
        results_file.flush()
        trans_file.flush()

    assert bleu_score > 10, "BLEU score is too low!"
    assert comet_score > 0.5, "COMET score is too low!"
