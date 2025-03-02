import os
import csv
import pytest
import random
from datasets import load_dataset
from transformers import M2M100ForConditionalGeneration
from helpers.tokenization_small100 import SMALL100Tokenizer
from helpers.evaluation import compute_bleu, compute_comet  # Ensure helpers are correct

# Define the path for the results CSV file
RESULTS_CSV = "data/small100_test_results.csv"

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

WMT_DATASET = "google/wmt24pp"

def write_to_csv(model, language, bleu, comet):
    """Append a row to the CSV file."""
    file_exists = os.path.isfile(RESULTS_CSV)
    
    with open(RESULTS_CSV, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Model", "Language", "Bleu", "Comet"])  # Write header if file does not exist
        writer.writerow([model, language, bleu, comet])

def load_wmt_data(target_lang_code):
    """Load and shuffle WMT test data for Small100."""
    dataset_name = f"en-{LANGUAGES_TO_TEST[target_lang_code]}"

    print(f"🔎 Loading WMT dataset: {dataset_name}")

    try:
        dataset = load_dataset(WMT_DATASET, dataset_name)
    except ValueError:
        print(f" No dataset found for {dataset_name}. Skipping...")
        write_to_csv("alirezamsh/small100", target_lang_code, "NA", "NA")
        pytest.skip(f"Skipping {dataset_name}: No dataset found.")

    if "train" in dataset and len(dataset["train"]) > 0:
        split_name = "train"
    else:
        print(f" No usable split found for {dataset_name}. Skipping...")
        write_to_csv("alirezamsh/small100", target_lang_code, "NA", "NA")
        pytest.skip(f"Skipping {dataset_name}: No usable split found.")

    test_samples = list(dataset[split_name])

    if not test_samples:
        print(f" No test samples found for {dataset_name}. Skipping...")
        write_to_csv("alirezamsh/small100", target_lang_code, "NA", "NA")
        pytest.skip(f"Skipping {dataset_name}: No test samples found.")

    # Shuffle dataset for randomness
    random.seed(42)  # Ensures reproducibility of shuffle order
    random.shuffle(test_samples)
    test_samples = test_samples[:5]  # Take 5 shuffled samples

    sources = [sample["source"] for sample in test_samples]
    references = [sample["target"] for sample in test_samples]  # Use post-edit as reference

    return sources, references

@pytest.mark.parametrize("target_lang_code", LANGUAGES_TO_TEST.keys())
def test_translation_quality(target_lang_code, request):
    """Test Small100 translations using WMT data and log results in CSV."""
    print(f"🛠️ Loading Small100 model for {target_lang_code}...")

    # Load Small100 model
    model = M2M100ForConditionalGeneration.from_pretrained("alirezamsh/small100")
    tokenizer = SMALL100Tokenizer.from_pretrained("alirezamsh/small100", tgt_lang=target_lang_code)

    sources, references = load_wmt_data(target_lang_code)

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

    assert bleu_score > 10, "BLEU score is too low!"
    assert comet_score > 0.5, "COMET score is too low!"
