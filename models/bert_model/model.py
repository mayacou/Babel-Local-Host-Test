from transformers import MBartForConditionalGeneration, MBart50Tokenizer
import torch

def bert_load(model_name="facebook/mbart-large-50-many-to-many-mmt"):
    # Load the mBART tokenizer and model
    tokenizer = MBart50Tokenizer.from_pretrained(model_name)
    model = MBartForConditionalGeneration.from_pretrained(model_name)

    return model, tokenizer
import torch

# List of valid language codes
LANGUAGE_MAP = {
    "ar": "ar_AR", "cs": "cs_CZ", "de": "de_DE", "en": "en_XX", "es": "es_XX", "et": "et_EE", 
    "fi": "fi_FI", "fr": "fr_XX", "gu": "gu_IN", "hi": "hi_IN", "it": "it_IT", "ja": "ja_XX", 
    "kk": "kk_KZ", "ko": "ko_KR", "lt": "lt_LT", "lv": "lv_LV", "my": "my_MM", "ne": "ne_NP", 
    "nl": "nl_XX", "ro": "ro_RO", "ru": "ru_RU", "si": "si_LK", "tr": "tr_TR", "vi": "vi_VN", 
    "zh": "zh_CN", "af": "af_ZA", "az": "az_AZ", "bn": "bn_IN", "fa": "fa_IR", "he": "he_IL", 
    "hr": "hr_HR", "id": "id_ID", "ka": "ka_GE", "km": "km_KH", "mk": "mk_MK", "ml": "ml_IN", 
    "mn": "mn_MN", "mr": "mr_IN", "pl": "pl_PL", "ps": "ps_AF", "pt": "pt_XX", "sv": "sv_SE", 
    "sw": "sw_KE", "ta": "ta_IN", "te": "te_IN", "th": "th_TH", "tl": "tl_XX", "uk": "uk_UA", 
    "ur": "ur_PK", "xh": "xh_ZA", "gl": "gl_ES", "sl": "sl_SI"
}

def fix_lang_code(lang):
    lang_prefix = lang[:2]  # Extract the first two letters

    if lang in LANGUAGE_MAP.values():
        return lang  # Already correct
    
    if lang_prefix in LANGUAGE_MAP:
        return LANGUAGE_MAP[lang_prefix]  # Fix it

    raise ValueError(f"Unsupported language code: {lang}")

def bert_inference(test_data, model, tokenizer, src_lang, tgt_lang, config=None, debug=True):
    config = config or {"BEAM_SIZE": 5, "LENGTH_PENALTY": 1.2, "MAX_LENGTH": 400}
    generated_translations = []

    # Fix src_lang and tgt_lang format
    try:
        src_lang = fix_lang_code(src_lang)
        tgt_lang = fix_lang_code(tgt_lang)
    except ValueError as e:
        if debug:
            print(f"Language Error: {e}")
        return [""] * len(test_data)  # Return empty translations if language is invalid

    if debug:
        print(f"Using src_lang: {src_lang}, tgt_lang: {tgt_lang}")

    # Set tokenizer source language safely
    try:
        tokenizer.src_lang = src_lang
    except Exception as e:
        if debug:
            print(f"Error setting tokenizer.src_lang: {e}")
        tokenizer.src_lang = None  # Handle safely

    # Convert target language to token ID safely
    try:
        forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_lang)
    except Exception as e:
        if debug:
            print(f"Error converting tgt_lang to token ID: {e}")
        forced_bos_token_id = None  # Handle safely

    for example in test_data:
        input_text = example if isinstance(example, str) else example.get('source', '')

        if not input_text:
            generated_translations.append("")
            continue

        try:
            prompt = f"{input_text}"

            inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(model.device)

            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    num_beams=config["BEAM_SIZE"],
                    length_penalty=config["LENGTH_PENALTY"],
                    early_stopping=False,
                    forced_bos_token_id=forced_bos_token_id,
                    max_length=config["MAX_LENGTH"]
                )

            if output is None or output.size(0) == 0:
                generated_translations.append("")
                continue

            decoded_output = tokenizer.batch_decode(output, skip_special_tokens=True)

            if decoded_output:
                generated_translation = decoded_output[0]
                generated_translations.append(generated_translation.strip())
            else:
                generated_translations.append("")

        except Exception as e:
            if debug:
                print(f"Error during inference: {e}")
            generated_translations.append("")

    return generated_translations
