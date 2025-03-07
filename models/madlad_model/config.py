import torch

MODEL_NAME = "jbochi/madlad400-7b-mt"
QUANTIZATION_CONFIG = None  # Adjust if using quantization
ATTN_IMPLEMENTATION = "default"  # Modify if needed
DEVICE_MAP = "auto"  # Change based on GPU/CPU setup
TORCH_DTYPE = torch.float16
