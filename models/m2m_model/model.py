# m2m_model/model.py
# No config file needed for now

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from .config import MODEL_NAME

def m2m_load():
    """Load the M2M-100 model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    return model, tokenizer