import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from .config import MODEL_NAME

def towerinstruct_load():
    """
    Load the TowerInstruct model and tokenizer and automatically use GPU if available.
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
        
        # Move the model to GPU if available
        if torch.cuda.is_available():
            model = model.to("cuda")
        else:
            model = model.to("cpu")
        
        return model, tokenizer
    except Exception as e:
        print(f"❌ Error loading {MODEL_NAME}: {e}")
        return None, None
