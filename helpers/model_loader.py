from transformers import MarianMTModel, MarianTokenizer

def load_model(model_name):
    """Load a translation model and tokenizer from Hugging Face."""
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return model, tokenizer

def translate_text(model, tokenizer, text, target_lang, max_length=512, max_new_tokens=150):
    """
    Translates text using the given model and tokenizer.
    Ensures correct target language selection by adding language tokens for multilingual models.
    """
    # If model is "Helsinki-NLP/opus-mt-en-sla", prepend language token
    MULTILINGUAL_MODELS = ["Helsinki-NLP/opus-mt-en-sla"]
    LANGUAGE_TOKENS = {"hr": ">>hrv<<", "pl": ">>pol<<", "sl": ">>slv<<"}

    if model.config._name_or_path in MULTILINGUAL_MODELS:
        lang_token = LANGUAGE_TOKENS.get(target_lang, None)
        if lang_token:
            text = f"{lang_token} {text}"  # Add token to input

    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)

    # Generate translation with adjusted decoding parameters
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        max_new_tokens=max_new_tokens,
        num_beams=5,  # Encourages more complete translations
        length_penalty=1.2,  # Prevents overly short translations
        early_stopping=False
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


