# models/mistral/mistral_model.py

from transformers import AutoModelForCausalLM, AutoTokenizer
from models.mistral_model.config import MODEL_NAME, QUANTIZATION_CONFIG, ATTN_IMPLEMENTATION, DEVICE_MAP, TORCH_DTYPE

def mistral_load(model_name=MODEL_NAME, 
                             quantization_config=QUANTIZATION_CONFIG, 
                             attn_implementation=ATTN_IMPLEMENTATION, 
                             device_map=DEVICE_MAP,
                             torch_dtype=TORCH_DTYPE
                             ):
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Load the model with the configuration
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device_map,
        # quantization_config=quantization_config,
        attn_implementation=attn_implementation
    )
    
    return model, tokenizer

