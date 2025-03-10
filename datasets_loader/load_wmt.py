import random
from datasets import load_dataset

# Mapping of available WMT24PP language pairs (full locale names, change es_MX to es-ES if it exists)
LANGUAGE_CODE_MAP = {
    "bg": "bg_BG", "cs": "cs_CZ",
    "da": "da_DK", "de": "de_DE", "el": "el_GR", "es": "es_ES", "et": "et_EE",
    "fi": "fi_FI", "fr": "fr_FR", "hu": "hu_HU",
    "is": "is_IS", "it": "it_IT", "lt": "lt_LT", "lv": "lv_LV",
    "nl": "nl_NL", "pl": "pl_PL", "pt": "pt_PT", "ro": "ro_RO",
    "sk": "sk_SK", "sv": "sv_SE", "tr": "tr_TR", "hr": "hr_HR"
}

def load_wmt_data(language_pair):
    """
    Load and shuffle test data for the given language pair from the WMT dataset.
    If "get_languages" is passed, return the list of available language pairs.
    """
    if language_pair == "get_languages":
        return list(LANGUAGE_CODE_MAP.keys())  # Return mapped full region codes

    # Ensure language_pair is mapped to full region format if needed
    mapped_lang_pair = LANGUAGE_CODE_MAP.get(language_pair, language_pair)
    dataset_name = f"en-{mapped_lang_pair}"  # Adjusted dataset name

    print(f"🔎 Attempting to load dataset: {dataset_name}")  # Debugging output

    try:
        dataset = load_dataset("google/wmt24pp", dataset_name)
    except ValueError:
        print(f"❌ Dataset {dataset_name} not found. Skipping...")
        return [], []

    if "train" in dataset and len(dataset["train"]) > 0:
        split = "train"
    else:
        print(f"❌ No usable split found for {dataset_name}. Skipping...")
        return [], []

    test_samples = list(dataset[split])

    if not test_samples:
        print(f"❌ No test samples found for {dataset_name}. Skipping...")
        return [], []

    # Shuffle dataset for randomness
    random.seed(42)  # Ensures reproducibility
    random.shuffle(test_samples)
    test_samples = test_samples[:1]  # Take 5 shuffled samples

    sources = [sample["source"] for sample in test_samples]
    references = [sample["target"] for sample in test_samples]

    return sources, references
