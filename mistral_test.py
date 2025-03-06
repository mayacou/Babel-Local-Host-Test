import logging
import os
from evaluations.metrics import compute_bleu, compute_comet
from scripts.data_loader import load_europarl_data
from scripts.csv_helpers import write_to_csv
from config.languages import EUROPARL_LANG_PAIRS
from models.load_model import load_model_and_tokenizer
from models.mistral_model.model import perform_inference

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
    first_language_name = language_pair[0]
    second_language_name = language_pair[1]

    print(f"Running test for {first_language_name} -> {second_language_name}...")

    # Load the data (en → target_lang_code)
    sources, references = load_europarl_data(language_pair[1])
    if not sources or not references:
        print(f"⚠️ Skipping {language_pair}: No dataset found.")
        return -1, -1

    # Normalize references
    references = [ref.strip().lower() for ref in references]

    # Load model and tokenizer (for example, using a "mistral" identifier)
    model, tokenizer = load_model_and_tokenizer("mistral")

    # Use perform_inference to generate hypotheses
    # perform_inference should take the list of source sentences along with the model and tokenizer,
    # and return generated translations (hypotheses) and optionally reference texts.
    hypotheses, _ = perform_inference(sources, model, tokenizer)
    
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

    # Optionally, log all scores
    logging.info("\nAll scores: ")
    logging.info(all_scores)
    print("All scores logged to mistral_test_results.log")

if __name__ == "__main__":
    main()
