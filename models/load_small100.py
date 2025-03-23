# load_small100.py
import torch
from transformers import M2M100ForConditionalGeneration
from helpers.tokenization_small100 import SMALL100Tokenizer

def load_small100(target_lang_code):
    model_name = "alirezamsh/small100"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = M2M100ForConditionalGeneration.from_pretrained(model_name).to(device)
    tokenizer = SMALL100Tokenizer.from_pretrained(model_name, tgt_lang=target_lang_code)
    
    return model, tokenizer, device
