import torch
import torch_directml
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

#device = torch_directml.device(torch_directml.default_device())

def load_model():
    """Load the NLLB-200 model and tokenizer."""
    model_name = "facebook/nllb-200-distilled-600M"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)#.to(device)
    print("Model loaded successfully!")
    return model, tokenizer

def translate_text(model, tokenizer, text, src_lang="eng_Latn", tgt_lang="spa_Latn"):
    """Translate text using NLLB-200 model."""
    tokenizer.src_lang = src_lang
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)#.to(device)
    forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_lang)
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        max_new_tokens=max_new_tokens,
        num_beams=5,  # Encourages more complete translations
        length_penalty=1.2,  # Prevents overly short translations
        early_stopping=False,
        forced_bos_token_id=forced_bos_token_id
    )
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]