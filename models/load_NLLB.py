import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def load_model():
    """Load the NLLB-200 model and tokenizer with GPU support."""
    model_name = "facebook/nllb-200-distilled-600M"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Detect available device
    if torch.cuda.is_available():
        device = torch.device("cuda")  # NVIDIA GPU
        print("🚀 Using NVIDIA GPU (CUDA)")
    else:
        device = torch.device("cpu")  # Default to CPU
        print("⚠️ Using CPU (No GPU detected)")

    # Load model onto the selected device
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    
    print("✅ NLLB-200 Model loaded successfully!")
    return model, tokenizer, device

def translate_text(model, tokenizer, text, src_lang="eng_Latn", tgt_lang="spa_Latn"):
    """Translate text using NLLB-200 model."""
    tokenizer.src_lang = src_lang
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
    forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_lang)
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