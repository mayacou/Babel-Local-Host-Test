from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from models.madlad_model.config import MODEL_NAME, ATTN_IMPLEMENTATION, DEVICE_MAP, TORCH_DTYPE


def madlad_load(model_name=MODEL_NAME, 
                attn_implementation=ATTN_IMPLEMENTATION, 
                device_map=DEVICE_MAP,
                torch_dtype=TORCH_DTYPE):
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load the model with the correct class for seq2seq models
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device_map,
        attn_implementation=attn_implementation
    )
    
    return model, tokenizer

