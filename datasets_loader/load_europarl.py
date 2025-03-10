from datasets import load_dataset, get_dataset_config_names

BATCH_SIZE = 1

EUROPARL_LANG_PAIRS = [
    "bg", "cs", "da", "nl", "et", "fi", "fr", "de", "el", "hu", "it",
    "lv", "lt", "pl", "pt", "ro", "sk", "sl", "es", "sv", "tr"
]

def load_europarl_data(target_language):
    try:
        if target_language == "get_languages":
            return EUROPARL_LANG_PAIRS

        # Get available language pairs from dataset
        available_configs = get_dataset_config_names("Helsinki-NLP/europarl")

        # Check for 'en-XX' first, then 'XX-en'
        forward_pair = f"en-{target_language}"
        reverse_pair = f"{target_language}-en"

        if forward_pair in available_configs:
            langpair = forward_pair
            reverse_mode = False
        elif reverse_pair in available_configs:
            langpair = reverse_pair
            reverse_mode = True
        else:
            raise ValueError(f"❌ No dataset found for {target_language} in either direction.")

        print(f"🟢 Loading dataset: {langpair}\n")
        dataset = load_dataset("Helsinki-NLP/europarl", langpair, split="train", trust_remote_code=True).shuffle(seed=42)

        source_sentences = []
        reference_sentences = []

        for item in dataset.select(range(BATCH_SIZE)):
            if "translation" in item:
                translation = item["translation"]

                # Extract correct language codes
                src_lang, tgt_lang = langpair.split("-")

                if reverse_mode:
                    # Swap for XX-en datasets
                    source_sentences.append(translation[tgt_lang])  # English text
                    reference_sentences.append(translation[src_lang])  # Target language text
                else:
                    # Normal case: en-XX
                    source_sentences.append(translation[src_lang])  # English text
                    reference_sentences.append(translation[tgt_lang])  # Target language text

        return source_sentences, reference_sentences

    except Exception as e:
        print(f"❌ Error loading dataset for {target_language}: {e}")
        return [], []
