import logging
import os
from transformers import MistralForConditionalGeneration, MistralTokenizer
from evaluations.metrics import compute_bleu, compute_comet
from scripts.data_loader import load_europarl_data
from scripts.csv_helpers import write_to_csv
from config.languages import EUROPARL_LANG_PAIRS
from models.load_model import load_model_and_tokenizer

# Define the path for the results CSV file
RESULTS_CSV = "data/mistral_test_results.csv"

# Logging Setup
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "mistral_test_results.log")

logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def run_test_for_language_pair(language_pair):
    first_language_name = EUROPARL_LANG_PAIRS.get(language_pair[0], "Unknown Language")
    second_language_name = EUROPARL_LANG_PAIRS.get(language_pair[1], "Unknown Language")

    print(f"Running test for {first_language_name} -> {second_language_name}...")

    # Load the data (en → target_lang_code)
    sources, references = load_europarl_data(language_pair[1])
    if not sources or not references:
        print(f"⚠️ Skipping {language_pair}: No dataset found.")
        return -1, -1

    # Normalize references
    references = [ref.strip().lower() for ref in references]

    # Initialize hypotheses list
    hypotheses = []

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer("mistral-7b")

    for sentence in sources:
        model_inputs = tokenizer(sentence.strip(), return_tensors="pt", padding=True, truncation=True, max_length=256)
        output_tokens = model.generate(**model_inputs, num_beams=10, max_length=256, early_stopping=True)
        hypothesis = tokenizer.batch_decode(output_tokens, skip_special_tokens=True)[0].strip().lower()
        hypotheses.append(hypothesis)

    # Evaluate translations using BLEU and COMET
    bleu_score = compute_bleu(references, hypotheses)
    comet_score = compute_comet(references, hypotheses, sources)

    print(f"BLEU Score: {bleu_score}")
    print(f"COMET Score: {comet_score}")

    # Log results to CSV
    write_to_csv("mistral-7b", language_pair[1], round(bleu_score, 2), round(comet_score, 2))

    return bleu_score, comet_score

def main():
    all_scores = {}
    language_pairs = [('en', code) for code in EUROPARL_LANG_PAIRS]

    for language_pair in language_pairs:
        try:
            bleu_score, comet_score = run_test_for_language_pair(language_pair)
            all_scores[language_pair] = {
                "BLEU": bleu_score,
                "COMET": comet_score
            }

            # Log each language pair result
            logging.info(f"Test results for {language_pair[0]} -> {language_pair[1]}:")
            logging.info(f"  BLEU Score: {bleu_score}")
            logging.info(f"  COMET Score: {comet_score}")
            logging.info("-" * 40)

        except Exception as e:
            logging.error(f"Error testing {language_pair}: {str(e)}")

    # Optionally, save or log results
    logging.info("\nAll scores: ")
    logging.info(all_scores)

    # Also print final summary to console
    print("All scores logged to mistral_test_results.log")

if __name__ == "__main__":
    main()
