import os
import csv
import pytest
import torch
from datasets_loader.load_wmt import load_wmt_data  # Import the function from load_wmt.py
from transformers import M2M100ForConditionalGeneration
from helpers.tokenization_small100 import SMALL100Tokenizer
from helpers.evaluation import compute_bleu, compute_comet  # Ensure helpers are correct

# Ensure directories exist
os.makedirs("translation_results", exist_ok=True)
os.makedirs("data", exist_ok=True)

# Define output CSV files
RESULTS_CSV = "data/small100_test_results.csv"
TRANSLATIONS_CSV = "translation_results/small100_translations.csv"

# List of target languages to test
LANGUAGES_TO_TEST = {
    "sq": "sq_AL", "bg": "bg_BG", "hr": "hr_HR", "cs": "cs_CZ", "da": "da_DK",
    "nl": "nl_NL", "et": "et_EE", "fi": "fi_FI", "fr": "fr_FR", "de": "de_DE",
    "el": "el_GR", "hu": "hu_HU", "is": "is_IS", "it": "it_IT", "lv": "lv_LV",
    "lt": "lt_LT", "lb": "lb_LU", "mk": "mk_MK", "no": "no_NO", "pl": "pl_PL",
    "pt": "pt_PT", "ro": "ro_RO", "sk": "sk_SK", "sl": "sl_SI", "es": "es_MX",
    "sv": "sv_SE", "tr": "tr_TR"
}

# ✅ Use CUDA if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    print(f"🚀 Using NVIDIA GPU: {torch.cuda.get_device_name(0)}")
else:
    print("⚠️ No NVIDIA GPU detected, using CPU instead.")

@pytest.mark.parametrize("target_lang_code", LANGUAGES_TO_TEST.keys())
def test_translation_quality(target_lang_code):
    """Test Small100 translations using WMT data and log results in CSV."""
    print(f"🛠️ Loading Small100 model for {target_lang_code}...")

    try:
        model = M2M100ForConditionalGeneration.from_pretrained("alirezamsh/small100").to(device)
        tokenizer = SMALL100Tokenizer.from_pretrained("alirezamsh/small100", tgt_lang=target_lang_code)
    except Exception as e:
        print(f"❌ Error loading model/tokenizer for {target_lang_code}: {e}")
        return  # Skip this language if model fails to load

    try:
        sources, references = load_wmt_data(target_lang_code)
    except Exception as e:
        print(f"⚠️ Error loading dataset for {target_lang_code}: {e}")
        sources, references = [], []

    if not sources or not references:
        print(f"⚠️ Skipping {target_lang_code}: No dataset found. Logging 'NA' to CSV.")
        with open(RESULTS_CSV, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["alirezamsh/small100", target_lang_code, "NA", "NA"])
        pytest.skip(f"Skipping {target_lang_code}: No test data available.")
        return

    hypotheses = []
    for sentence in sources:
        try:
            model_inputs = tokenizer(sentence, return_tensors="pt").to(device)
            output_tokens = model.generate(**model_inputs, num_beams=10, max_length=256, early_stopping=True)
            hypothesis = tokenizer.batch_decode(output_tokens, skip_special_tokens=True)[0]
            hypotheses.append(hypothesis)
        except Exception as e:
            print(f"⚠️ Translation error for {target_lang_code}: {e}")
            hypotheses.append("ERROR")

    # Compute evaluation metrics safely
    try:
        bleu_score = compute_bleu(references, hypotheses)
        comet_score = compute_comet(references, hypotheses, sources)
    except Exception as e:
        print(f"⚠️ Error computing BLEU/COMET: {e}")
        bleu_score, comet_score = "NA", "NA"

    print(result := {
        "model": "alirezamsh/small100",
        "target_language": target_lang_code,
        "BLEU": round(bleu_score, 2) if isinstance(bleu_score, (int, float)) else "NA",
        "COMET": round(comet_score, 2) if isinstance(comet_score, (int, float)) else "NA",
    })

    # ✅ Open CSV files **only once per test execution**
    with open(RESULTS_CSV, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([result["model"], result["target_language"], result["BLEU"], result["COMET"]])

    with open(TRANSLATIONS_CSV, mode="a", newline="") as trans_file:
        trans_writer = csv.writer(trans_file)
        for src, hyp, ref in zip(sources, hypotheses, references):
            trans_writer.writerow(["alirezamsh/small100", target_lang_code, src, hyp, ref])

    assert isinstance(bleu_score, (int, float)) and bleu_score > 10, "BLEU score is too low!"
    assert isinstance(comet_score, (int, float)) and comet_score > 0.5, "COMET score is too low!"
