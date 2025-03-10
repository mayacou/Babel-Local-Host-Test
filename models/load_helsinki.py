from transformers import MarianMTModel, MarianTokenizer

def load_model(model_name):
   """Load a translation model and tokenizer from Hugging Face."""
   tokenizer = MarianTokenizer.from_pretrained(model_name)
   model = MarianMTModel.from_pretrained(model_name)
   return model, tokenizer

def translate_text(model, tokenizer, text):
   """Translate text using the specified model."""
   inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
   translated = model.generate(
      **inputs,
      # max_length=max_length,
      # max_new_tokens=max_new_tokens,
      num_beams=5,  # Encourages more complete translations
      length_penalty=1.2,  # Prevents overly short translations
      early_stopping=False
   )
   return tokenizer.decode(translated[0], skip_special_tokens=True)