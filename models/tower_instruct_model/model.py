from transformers import AutoModelForCausalLM, AutoTokenizer
from .config import MODEL_NAME

def towerinstruct_load():
    """
    Load the TowerInstruct model and tokenizer.
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
        return model, tokenizer
    except Exception as e:
        print(f"❌ Error loading {MODEL_NAME}: {e}")
        return None, None