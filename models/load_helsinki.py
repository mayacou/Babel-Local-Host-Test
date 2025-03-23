import torch
from transformers import MarianMTModel, MarianTokenizer

def load_model(model_name):
    """Load a translation model and tokenizer from Hugging Face and move model to GPU if available."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name).to(device)  # Move model to GPU

    return model, tokenizer, device  # Return device for later use


def translate_text(model, tokenizer, text, device):
   """Translate text using the specified model."""
   inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
   translated = model.generate(
      **inputs,
      # max_length=max_length,
      # max_new_tokens=max_new_tokens,
      num_beams=5,  # Encourages more complete translations
      length_penalty=1.2,  # Prevents overly short translations
      early_stopping=False
   )
   return tokenizer.decode(translated[0], skip_special_tokens=True)