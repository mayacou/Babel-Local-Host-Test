import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_towerinstruct(parameters):
    """
    Load the TowerInstruct model based on user input (7 for 7B, 13 for 13B) with GPU support.
    """
    model_name = None

    if parameters == 7:
        model_name = "Unbabel/TowerInstruct-7B-v0.2"
    elif parameters == 13:
        model_name = "Unbabel/TowerInstruct-13B-v0.1"
    else:
        print("❌ Incorrect parameter passed! Use 7 or 13.")
        return None, None, None

    print(f"🔄 Loading {model_name} model...")

    # Detect available device
    if torch.cuda.is_available():
        device = torch.device("cuda")  # NVIDIA GPU
        print("🚀 Using NVIDIA GPU (CUDA)")
    else:
        device = torch.device("cpu")  # Default to CPU
        print("⚠️ Using CPU (No GPU detected)")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)  # Load model on GPU
        print(f"✅ Successfully loaded {model_name}!")
        return model, tokenizer, device
    except Exception as e:
        print(f"❌ Error loading {model_name}: {e}")
        return None, None, None

def translate_text(model, tokenizer, text):
    """
    Translate text using the specified TowerInstruct model.
    """
    if model is None or tokenizer is None:
        print("❌ Model or tokenizer is not loaded.")
        return ""
    
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



