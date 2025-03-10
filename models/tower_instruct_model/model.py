import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from .config import MODEL_NAME

def towerinstruct_load():
    """
    Load the TowerInstruct model and tokenizer on GPU if available.
    """
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
        return model, tokenizer, device
    except Exception as e:
        print(f"❌ Error loading {MODEL_NAME}: {e}")
        return None, None, None
