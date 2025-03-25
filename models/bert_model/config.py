# config.py
MODEL_NAME = 'bert-base-multilingual-cased'  # Replace with your desired model for multi-language tasks
TOKENIZER_NAME = 'bert-base-multilingual-cased'  # Ensure tokenizer matches the model
DEVICE_MAP = 'auto'  # Change based on your hardware setup (e.g., 'cuda', 'cpu', 'auto')
TORCH_DTYPE = 'float32'  # Data type for model weights
PAD_TOKEN = '[PAD]'  # The padding token, you can customize this if needed
