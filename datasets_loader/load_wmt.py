import json
import re
from datasets import load_dataset

# Mapping of available WMT24PP language pairs
WMT24PP_LANGUAGE_PAIRS = [
    "ar_EG", "ar_SA", "bg_BG", "bn_IN", "ca_ES", "cs_CZ", "da_DK", "de_DE",
    "el_GR", "es_MX", "et_EE", "fa_IR", "fi_FI", "fr_CA", "fr_FR", "gu_IN", "he_IL", 
    "hi_IN", "hr_HR", "hu_HU", "id_ID", "is_IS", "it_IT", "ja_JP", "kn_IN", "ko_KR", 
    "lt_LT", "lv_LV", "ml_IN", "mr_IN", "nl_NL", "no_NO", "pa_IN", "pl_PL", "pt_BR", 
    "pt_PT", "ro_RO", "ru_RU", "sk_SK", "sl_SI", "sr_RS", "sv_SE", "sw_KE", "sw_TZ", 
    "ta_IN", "te_IN", "th_TH", "tr_TR", "uk_UA", "ur_PK", "vi_VN", "zh_CN", "zh_TW"
]

def load_wmt_data(language_pair):
    """
    Load test data for the given language pair from the WMT dataset.
    If "get_languages" is passed, return the list of available language pairs.
    """
    if language_pair == "get_languages":
        return WMT24PP_LANGUAGE_PAIRS  # Return available language pairs
    
    dataset_name = f"en-{language_pair}"  # Adjust dataset naming convention if needed
    
    try:
        dataset = load_dataset("google/wmt24pp", dataset_name)
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

