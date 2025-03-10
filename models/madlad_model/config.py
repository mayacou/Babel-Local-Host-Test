import torch

# Model Configuration
MODEL_NAME = "google/madlad400-7b-mt"  # MADLAD model name
ATTN_IMPLEMENTATION = "eager"  # Attention implementation
DEVICE_MAP = "auto"  # Automatically map model to available devices
TORCH_DTYPE = torch.float16  # Data type for model weights (float16 for GPU efficiency)

# Tokenizer Configuration
TOKENIZER_NAME = MODEL_NAME  # Typically the same as the model name
PAD_TOKEN = "<pad>"  # Padding token (if not already set in the tokenizer)
MAX_LENGTH = 512  # Maximum sequence length for tokenization

# Inference Configuration
# MAX_GENERATION_LENGTH = 60 # Maximum length for generated translations
BEAM_SIZE = 5  # Beam size for beam search
TEMPERATURE = 1.0  # Sampling temperature
TOP_K = 50  # Top-k sampling
TOP_P = 0.95  # Top-p (nucleus) sampling