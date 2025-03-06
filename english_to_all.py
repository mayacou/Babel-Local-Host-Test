import logging
import torch
from scripts.run_tests import run_test_for_language_pair
from config.languages import LANGUAGES
import os

# Set up logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "en_to_all_test_results.log")

logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def main():
    all_scores = {}
    language_pairs = [('en', code) for code in LANGUAGES.keys()]

    for language_pair in language_pairs:
        try:
            bleu_score, comet_score = run_test_for_language_pair(language_pair)
            all_scores[language_pair] = {
                "BLEU": bleu_score,
                "COMET": comet_score
            }

            # Log each language pair result
            language_from = LANGUAGES.get(language_pair[0], "Unknown Language")
            language_to = LANGUAGES.get(language_pair[1], "Unknown Language")
            logging.info(f"Test results for {language_from} -> {language_to}:")
            logging.info(f"  BLEU Score: {bleu_score}")
            logging.info(f"  COMET Score: {comet_score}")
            logging.info("-" * 40)  
        
        except Exception as e:
            logging.error(f"Error testing {language_pair}: {str(e)}")
        
        finally:
            # Clear GPU memory after each run
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
    # Optionally, save or log results
    logging.info("\nAll scores: ")
    logging.info(all_scores)

    # Also print final summary to console
    print("All scores logged to en_to_all_test_results.log")

if __name__ == "__main__":
    main()
