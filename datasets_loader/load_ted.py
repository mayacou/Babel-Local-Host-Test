import json
import re
from datasets import load_dataset

TED_LANG_PAIRS = [
    "bg", "cs", "da", "nl", "et", "fi", "fr", "de", "el", "hr", "hu", "is", "it",
    "lv", "lt", "mk", "pl", "pt", "ro", "sk", "sl", "sq", "es", "sv", "tr"
]

def load_ted_data(language_pair):
    """
    Load test data for the given language pair from the TED Talk dataset.
    """
    if language_pair == "get_languages":
        return TED_LANG_PAIRS
    
    try:
        dataset = load_dataset("Helsinki-NLP/ted_talks_iwslt", language_pair)
    except ValueError:
        print(f"⚠️ Skipping {language_pair}: No dataset found.")
        return [], []

    if "train" in dataset:
        split = "train"
    else:
        print(f"⚠️ Skipping {language_pair}: No usable split found.")
        return [], []

    test_samples = list(dataset[split])[:5]  # Take 5 samples
    sources = [sample["source"] for sample in test_samples]
    references = [sample["target"] for sample in test_samples]
    
    return sources, references