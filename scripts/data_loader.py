# scripts/data_loader.py

from datasets import load_dataset

def load_opus_data(language_pair):
    try:
        # OPUS supports many languages; you can load specific language pairs.
        dataset = load_dataset("opus100", f"{language_pair[0]}-{language_pair[1]}")
        return dataset
    except ValueError:
        print(f"Language pair {language_pair[0]}-{language_pair[1]} not found in OPUS.")
        return None


def load_wmt_data(language_pair, sample_size=150):
    # Load the WMT dataset for a specific language pair
    dataset = load_dataset("wmt14", f"{language_pair[0]}-{language_pair[1]}")
    
    # Select a sample of the test data
    test_data = dataset['test'].select(range(sample_size)) 
    return test_data
