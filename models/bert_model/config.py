# config.py
MODEL_NAME = "facebook/mbart-large-50-many-to-many-mmt"  # Replace with your desired model for multi-language tasks
TOKENIZER_NAME = MODEL_NAME  # Ensure tokenizer matches the model
DEVICE_MAP = 'auto'  # Change based on your hardware setup (e.g., 'cuda', 'cpu', 'auto')
TORCH_DTYPE = 'float32'  # Data type for model weights
PAD_TOKEN = '[PAD]'  # The padding token, you can customize this if needed
