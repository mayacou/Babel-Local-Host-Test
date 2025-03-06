import logging
import os
from evaluations.metrics import compute_bleu, compute_comet
from scripts.csv_helpers import write_to_csv
from models.load_model import load_model_and_tokenizer
from models.mistral_model.model import perform_inference
from config.languages import get_dataset_loader, ALL_DATASETS

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

def run_test_for_language_pair(src_lang, tgt_lang, dataset_name):
    print(f"Running test for {src_lang} -> {tgt_lang} using {dataset_name} dataset...")

    # Get dataset loader function dynamically
    dataset_loader = get_dataset_loader(dataset_name)
    if not dataset_loader:
        print(f"⚠️ No dataset loader found for {dataset_name}. Skipping {tgt_lang}.")
        return -1, -1

    # Load the data
    try:
        sources, references = dataset_loader(src_lang, tgt_lang)
    except Exception as e:
        print(f"⚠️ Error loading dataset for {src_lang} -> {tgt_lang}: {e}")
        return -1, -1

    if not sources or not references:
        print(f"⚠️ Skipping {tgt_lang}: No dataset found in {dataset_name}.")
        return -1, -1

    references = [ref.strip().lower() for ref in references]

    # Load model and tokenizer
    try:
        model, tokenizer = load_model_and_tokenizer("mistral")
    except Exception as e:
        print(f"⚠️ Error loading model: {e}")
        return -1, -1

    # Perform inference
    try:
        hypotheses, _ = perform_inference(sources, model, tokenizer)
    except Exception as e:
        print(f"⚠️ Error during inference for {tgt_lang}: {e}")
        return -1, -1

    # Compute evaluation metrics
    bleu_score = compute_bleu(references, hypotheses)
    comet_score = compute_comet(references, hypotheses, sources)

    print(f"BLEU Score: {bleu_score}")
    print(f"COMET Score: {comet_score}")

    # Log results to CSV
    write_to_csv(RESULTS_CSV, "mistral", dataset_name ,tgt_lang, round(bleu_score, 2), round(comet_score, 2))

    # Free GPU memory
    del model
    del tokenizer
    import torch
    import gc
    torch.cuda.empty_cache()
    gc.collect()

    return bleu_score, comet_score

def main():
    all_scores = {}
    source_lang = "en"
    for dataset_name, language_pairs in ALL_DATASETS.items():
        for tgt_lang in language_pairs:
            try:
                bleu_score, comet_score = run_test_for_language_pair(source_lang, tgt_lang, dataset_name)
                all_scores[tgt_lang] = {
                    "BLEU": bleu_score,
                    "COMET": comet_score
                }

                logging.info(f"Test results for en -> {tgt_lang} using {dataset_name}:")
                logging.info(f"  BLEU Score: {bleu_score}")
                logging.info(f"  COMET Score: {comet_score}")
                logging.info("-" * 40)

            except Exception as e:
                logging.error(f"Error testing en -> {tgt_lang} with {dataset_name}: {str(e)}")

    logging.info("\nAll scores: ")
    logging.info(all_scores)
    print("All scores logged to mistral_test_results.log")

if __name__ == "__main__":
    main()
