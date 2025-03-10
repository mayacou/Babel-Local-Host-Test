from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from .config import MODEL_NAME

def nllb_load():
    """Load the NLLB-200 model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    return model, tokenizer