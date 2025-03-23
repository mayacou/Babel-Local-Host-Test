import json
import re
from datasets import load_dataset

def load_opus_data(language_pair):
    """
    Load test data for the given language pair from the OPUS dataset.
    """
    try:
        dataset = load_dataset("opus_books", language_pair)
    except ValueError:
        print(f"⚠️ Skipping {language_pair}: No dataset found.")
        return [], []

    if "train" in dataset:
        split = "train"
    else:
        print(f"⚠️ Skipping {language_pair}: No usable split found.")
        return [], []

    test_samples = list(dataset[split])[:1]  # Take 5 samples
    sources = [sample["source"] for sample in test_samples]
    references = [sample["target"] for sample in test_samples]
    
    return sources, references