from datasets import load_dataset

def load_opus_data(language_pair):
    """
    Load a specific language pair from the OPUS dataset.
    
    Args:
    - language_pair (tuple): A tuple containing the two language codes (e.g., ('en', 'it')).
    
    Returns:
    - dataset: Loaded dataset for the specified language pair, or None if not found.
    """
    try:
        # OPUS supports many languages; you can load specific language pairs.
        dataset = load_dataset("opus100", f"{language_pair[0]}-{language_pair[1]}")
        return dataset
    except ValueError:
        print(f"Language pair {language_pair[0]}-{language_pair[1]} not found in OPUS.")
        return None
    except Exception as e:
        print(f"An error occurred while loading the OPUS dataset: {e}")
        return None


def load_wmt_data(language_pair, sample_size=150):
    """
    Load a sample of the WMT dataset for a specific language pair.
    
    Args:
    - language_pair (tuple): A tuple containing the two language codes (e.g., ('en', 'de')).
    - sample_size (int): The number of samples to select from the test data (default is 150).
    
    Returns:
    - test_data: A subset of the test dataset for the specified language pair.
    """
    try:
        # Load the WMT dataset for a specific language pair
        dataset = load_dataset("wmt14", f"{language_pair[0]}-{language_pair[1]}")
        
        # Select a sample of the test data
        test_data = dataset['test'].select(range(sample_size)) 
        return test_data
    except Exception as e:
        print(f"An error occurred while loading the WMT dataset: {e}")
        return None
