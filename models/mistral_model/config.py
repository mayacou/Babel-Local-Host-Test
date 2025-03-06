# models/mistral/model_config.py

import torch
from transformers import BitsAndBytesConfig

# Configuration for the Mistral model and quantization
MODEL_NAME = "mistralai/Mistral-7B-v0.1"
QUANTIZATION_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)
ATTN_IMPLEMENTATION = "flash_attention_2"
DEVICE_MAP = "auto"
TORCH_DTYPE = torch.float16