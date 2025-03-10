import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def load_model():
    """Load the M2M-100 model and tokenizer with GPU support."""
    model_name = "facebook/m2m100_418M"  # Changed from NLLB-200
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Check available device: Use CUDA if available, otherwise DirectML (AMD) or CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("🚀 Using NVIDIA GPU (CUDA)")
    else:
        device = torch.device("cpu")
        print("⚠️ Using CPU (No GPU detected)")

    # Load model and move it to the selected device
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    
    print("✅ M2M-100 Model loaded successfully!")
    return model, tokenizer, device

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