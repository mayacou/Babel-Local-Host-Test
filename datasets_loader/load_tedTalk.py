import random
from datasets import load_dataset, get_dataset_config_names


TED_LANG_PAIRS = [ 
    "sq", "bg", "hr", "cs", "da", "nl", "et", "fi", "fr", "de", "el", "hu", "is", "it", 
    "lv", "lt", "mk", "nb", "pl", "pt", "ro", "sk", "sl", "es", "sv", "tr"
]


# Function to load TED Talk dataset
def load_tedTalk_data(target_lang_code, source_lang="en"):
    """
    Load TED Talk dataset for the specified target language.
    Uses the `davidstap/ted_talks` dataset from Hugging Face.
    """
    
    if target_lang_code == "get_languages":
        return TED_LANG_PAIRS

    # Define dataset name
    dataset_name = "davidstap/ted_talks"

    # Get available language pairs (must use trust_remote_code=True)
    available_configs = get_dataset_config_names(dataset_name, trust_remote_code=True)

    # Check if the requested language pair exists
    config_name = f"{source_lang}_{target_lang_code}"
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
    test_samples = test_samples[:5]  # Limit to 5 test samples

    # Extract source and target translations
    sources = [sample[source_lang] for sample in test_samples]
    references = [sample[target_lang_code] for sample in test_samples]

    return sources, references

# Example usage:
#source_lang = "en"
#target_lang = "fr"
#sources, references = load_tedTalk_data(target_lang, source_lang)
#for i, (src, ref) in enumerate(zip(sources, references), 1):
#    print(f"\n🔹 Sample {i}:\n🔸 Source ({source_lang}): {src}\n🔹 Reference ({target_lang}): {ref}")
