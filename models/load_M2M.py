import torch
import torch_directml
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def load_model():
    """Load the M2M-100 model and tokenizer."""
    model_name = "facebook/m2m100_418M"  # Changed from NLLB-200
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    print("M2M-100 Model loaded successfully!")
    return model, tokenizer

def translate_text(model, tokenizer, text, src_lang="en", tgt_lang="es"):
    """Translate text using M2M-100 model."""
    tokenizer.src_lang = src_lang
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    forced_bos_token_id = tokenizer.convert_tokens_to_ids(f"<<{tgt_lang}>>")  # Changed for M2M-100
    outputs = model.generate(
        **inputs,
        # max_length=max_length,
        # max_new_tokens=max_new_tokens,
        num_beams=5,  # Encourages more complete translations
        length_penalty=1.2,  # Prevents overly short translations
        early_stopping=False,
        forced_bos_token_id=forced_bos_token_id
    )
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]