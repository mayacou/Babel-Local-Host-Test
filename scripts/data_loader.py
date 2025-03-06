from datasets import load_dataset, get_dataset_config_names
import random
from config.languages import WMT_LANG_PAIRS, TED_LANG_PAIRS, EUROPARL_LANG_PAIRS

BATCH_SIZE = 10

def load_dataset_by_name(dataset_name, src_lang, trg_lang):
    """
    Load test data for the given source and target languages from the selected dataset.
    """
    print(f"Loading dataset: {dataset_name}, source language: {src_lang}, target language: {trg_lang}")

    if dataset_name == "WMT":
        return load_wmt_data(src_lang, trg_lang)
    elif dataset_name == "TED":
        return load_tedTalk_data(src_lang, trg_lang)
    elif dataset_name == "EUROPARL":
        return load_europarl_data(src_lang, trg_lang)
    elif dataset_name == "OPUS":
        return load_opus_data(src_lang, trg_lang)
    else:
        print(f"❌ Dataset {dataset_name} not recognized.")
        return [], []


def load_opus_data(src_lang, trg_lang):
    """
    Load test data for the given source and target languages from the OPUS dataset.
    """
    language_pair = f"{src_lang}-{trg_lang}"
    try:
        dataset = load_dataset("opus_books", language_pair)
    except ValueError:
        print(f"⚠️ Skipping {language_pair}: No dataset found.")
        return [], []

    split = "train" if "train" in dataset else None
    if not split:
        print(f"⚠️ Skipping {language_pair}: No usable split found.")
        return [], []

    test_samples = list(dataset[split])[:BATCH_SIZE]  # Take BATCH_SIZE samples
    sources = [sample["source"] for sample in test_samples]
    references = [sample["target"] for sample in test_samples]
    
    return sources, references

def load_wmt_data(src_lang, trg_lang):
    """
    Load and shuffle test data for the given source and target languages from the WMT dataset.
    If "get_languages" is passed, return the list of available language pairs.
    """
    if src_lang == "get_languages":
        return list(WMT_LANG_PAIRS.keys())  # Return mapped full region codes

    # Ensure source and target languages are mapped to full region format if needed
    mapped_src_lang = WMT_LANG_PAIRS.get(src_lang, src_lang)
    mapped_trg_lang = WMT_LANG_PAIRS.get(trg_lang, trg_lang)
    dataset_name = f"{mapped_src_lang}-{mapped_trg_lang}"  # Adjusted dataset name

    print(f"🔎 Attempting to load dataset: {dataset_name}")  # Debugging output

    try:
        dataset = load_dataset("google/wmt24pp", dataset_name)
    except ValueError:
        print(f"❌ Dataset {dataset_name} not found. Skipping...")
        return [], []

    split = "train" if "train" in dataset and len(dataset["train"]) > 0 else None
    if not split:
        print(f"❌ No usable split found for {dataset_name}. Skipping...")
        return [], []

    test_samples = list(dataset[split])

    if not test_samples:
        print(f"❌ No test samples found for {dataset_name}. Skipping...")
        return [], []

    # Shuffle dataset for randomness
    random.seed(42)  # Ensures reproducibility
    random.shuffle(test_samples)
    test_samples = test_samples[:BATCH_SIZE]  # Take BATCH_SIZE shuffled samples

    sources = [sample["source"] for sample in test_samples]
    references = [sample["target"] for sample in test_samples]

    return sources, references

def load_tedTalk_data(src_lang, trg_lang):
    """
    Load TED Talk dataset for the specified source and target languages.
    Uses the `davidstap/ted_talks` dataset from Hugging Face.
    """
    
    # Define dataset name
    dataset_name = "davidstap/ted_talks"

    # Get available language pairs (must use trust_remote_code=True)
    available_configs = get_dataset_config_names(dataset_name, trust_remote_code=True)

    # Check if the requested language pair exists
    config_name = f"{src_lang}_{trg_lang}"
    if config_name not in available_configs:
        print(f"⚠️ Language pair '{config_name}' not found in TED Talks dataset.")
        print(f"ℹ️ Available pairs: {', '.join(available_configs[:10])}... (truncated)")
        return [], []

    print(f"🔎 Loading TED dataset: {dataset_name} ({config_name})")

    # Load dataset with remote code trust
    dataset = load_dataset(dataset_name, config_name, trust_remote_code=True)

    # Select the appropriate split
    split = "test" if "test" in dataset else "train"
    if split not in dataset:
        print(f"⚠️ No usable split found for {config_name}.")
        return [], []

    # Shuffle TED Talk data
    test_samples = list(dataset[split])
    random.seed(42)
    random.shuffle(test_samples)
    test_samples = test_samples[:BATCH_SIZE]  # Limit to BATCH_SIZE test samples

    # Extract source and target translations
    sources = [sample[src_lang] for sample in test_samples]
    references = [sample[trg_lang] for sample in test_samples]

    return sources, references

def load_europarl_data(src_lang, trg_lang):
    try:
        if src_lang == "get_languages":
            return EUROPARL_LANG_PAIRS

        # Get available language pairs from dataset
        available_configs = get_dataset_config_names("Helsinki-NLP/europarl")

        # Check for 'en-XX' first, then 'XX-en'
        forward_pair = f"{src_lang}-{trg_lang}"
        reverse_pair = f"{trg_lang}-{src_lang}"

        if forward_pair in available_configs:
            langpair = forward_pair
            reverse_mode = False
        elif reverse_pair in available_configs:
            langpair = reverse_pair
            reverse_mode = True
        else:
            raise ValueError(f"❌ No dataset found for {src_lang}-{trg_lang} in either direction.")

        print(f"🟢 Loading dataset: {langpair}\n")
        dataset = load_dataset("Helsinki-NLP/europarl", langpair, split="train", trust_remote_code=True).shuffle(seed=42)

        source_sentences = []
        reference_sentences = []

        for item in dataset.select(range(BATCH_SIZE)):
            if "translation" in item:
                translation = item["translation"]

                # Extract correct language codes
                src_lang_code, tgt_lang_code = langpair.split("-")

                if reverse_mode:
                    # Swap for XX-en datasets
                    source_sentences.append(translation[tgt_lang_code])  # English text
                    reference_sentences.append(translation[src_lang_code])  # Target language text
                else:
                    # Normal case: en-XX
                    source_sentences.append(translation[src_lang_code])  # English text
                    reference_sentences.append(translation[tgt_lang_code])  # Target language text

        return source_sentences, reference_sentences

    except Exception as e:
        print(f"❌ Error loading dataset for {src_lang}-{trg_lang}: {e}")
        return [], []
