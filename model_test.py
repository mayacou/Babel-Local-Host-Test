import logging
import os
import argparse
from evaluations.metrics import compute_bleu, compute_comet
from scripts.csv_helpers import write_to_csv
from models.load_model import load_model_and_tokenizer
from models.perform_inference import perform_inference
from config.languages import ALL_DATASETS
from scripts.data_loader import load_dataset_by_name
import torch
import gc

# Logging Setup
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "test_results.log")

logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def run_test_for_language_pair(model_name, model, tokenizer, src_lang, tgt_lang, dataset_name, results_csv):
    print(f"Running test for {src_lang} -> {tgt_lang} using {dataset_name} dataset with {model_name}...")

    dataset_loader = load_dataset_by_name(dataset_name, src_lang, tgt_lang)
    if not dataset_loader:
        print(f"⚠️ No dataset loader found for {dataset_name}. Skipping {tgt_lang}.")
        return -1, -1

    try:
        sources, references = dataset_loader
    except Exception as e:
        print(f"⚠️ Error loading dataset for {src_lang} -> {tgt_lang}: {e}")
        return -1, -1

    if not sources or not references:
        print(f"⚠️ Skipping {tgt_lang}: No dataset found in {dataset_name}.")
        return -1, -1

    references = [ref.strip().lower() for ref in references]

    # Perform inference
    try:
        hypotheses, _ = perform_inference(sources, model_name, model, tokenizer, target_language=tgt_lang)
    except Exception as e:
        print(f"⚠️ Error during inference for {tgt_lang}: {e}")
        return -1, -1

    # Compute evaluation metrics
    bleu_score = compute_bleu(references, hypotheses)
    comet_score = compute_comet(references, hypotheses, sources)

    print(f"BLEU Score: {bleu_score}")
    print(f"COMET Score: {comet_score}")

    # Log results to CSV
    write_to_csv(results_csv, model_name, dataset_name, tgt_lang, round(bleu_score, 2), round(comet_score, 2))

    return bleu_score, comet_score

def main():
    parser = argparse.ArgumentParser(description="Run translation model tests.")
    parser.add_argument("model", type=str, help="Model name to use for testing")
    args = parser.parse_args()

    model_name = args.model
    results_csv = f"data/test_results.csv" 

    # Load the model and tokenizer once
    try:
        model, tokenizer = load_model_and_tokenizer(model_name)
    except Exception as e:
        print(f"⚠️ Error loading model {model_name}: {e}")
        return

    all_scores = {}
    source_lang = "en"
    
    for dataset_name, language_pairs in ALL_DATASETS.items():
        for tgt_lang in language_pairs:
            try:
                bleu_score, comet_score = run_test_for_language_pair(
                    model_name, model, tokenizer, source_lang, tgt_lang, dataset_name, results_csv
                )
                all_scores[tgt_lang] = {"BLEU": bleu_score, "COMET": comet_score}

                logging.info(f"Test results for en -> {tgt_lang} using {dataset_name} with {model_name}:")
                logging.info(f"  BLEU Score: {bleu_score}")
                logging.info(f"  COMET Score: {comet_score}")
                logging.info("-" * 40)

            except Exception as e:
                logging.error(f"Error testing en -> {tgt_lang} with {dataset_name} using {model_name}: {str(e)}")

    logging.info("\nAll scores: ")
    logging.info(all_scores)
    print(f"All scores logged to test_results.log for model {model_name}")

    # Free GPU memory
    del model
    del tokenizer
    torch.cuda.empty_cache()
    gc.collect()

if __name__ == "__main__":
    main()
