from transformers import AutoModelForCausalLM, AutoTokenizer
from models.mistral_model.config import MODEL_NAME, ATTN_IMPLEMENTATION, DEVICE_MAP, TORCH_DTYPE
import torch
def mistral_load(model_name=MODEL_NAME, 
                 attn_implementation=ATTN_IMPLEMENTATION, 
                 device_map=DEVICE_MAP,
                 torch_dtype=TORCH_DTYPE):
    try:
        # Load the tokenizer
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token  # Set pad_token if it's missing
        torch.cuda.empty_cache()
        torch.cuda.memory_allocated()
        # Load the model with memory optimization settings
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map=device_map,
            attn_implementation=attn_implementation,
            low_cpu_mem_usage=True
        )
        print(torch.cuda.is_available())  # Should return True if CUDA is available
        print(next(model.parameters()).device)  # Should return 'cuda' if running on GPU
        print(torch.cuda.memory_allocated())  # Check used memory
        print(model.hf_device_map)  # Should show most layers on CUDA


        # Return model and tokenizer
        return model, tokenizer
    
    except Exception as e:
        print(f"Error loading model or tokenizer for {model_name}: {e}")
        return None, None
