from datasets import load_dataset

BATCH_SIZE = 5

EUROPARL_LANG_PAIRS = [
    "bg", "cs", "da", "nl", "et", "fi", "fr", "de", "el", "hu", "it",
    "lv", "lt", "pl", "pt", "ro", "sk", "sl", "es", "sv", "tr"
]

def load_europarl_data(language_pair):
    try:
        if language_pair == "get_languages":
            return EUROPARL_LANG_PAIRS
        
        langpair = f"en-{language_pair}"
        langpair = "-".join(sorted(langpair.split("-")))
        print(f"current langpair being passed into EuroParl: {langpair}\n")
        dataset = load_dataset("Helsinki-NLP/europarl", langpair, split="train")
        
        dataset = dataset.shuffle(seed=42)
        source_sentences = []
        reference_sentences = []
        for item in dataset.select(range(BATCH_SIZE)):  
            if "translation" in item and "en" in item["translation"] and language_pair in item["translation"]:
                source_sentences.append(item["translation"]["en"])
                reference_sentences.append(item["translation"][language_pair])
        return source_sentences, reference_sentences
    except Exception as e:
        print(f"Error loading dataset for {language_pair}: {e}")
        return [], []
