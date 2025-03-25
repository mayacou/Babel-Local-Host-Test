from transformers import AutoTokenizer, AutoModel
from models.bert_model.config import MODEL_NAME, TOKENIZER_NAME, DEVICE_MAP, TORCH_DTYPE, PAD_TOKEN

def bert_load():
    """
    Load a multilingual BERT model and tokenizer for multi-language tasks.

    Returns:
        model: Loaded multilingual BERT model.
        tokenizer: Loaded tokenizer.
    """
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    # Ensure the tokenizer has a padding token
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': PAD_TOKEN})

    # Load the model with from_tf=True to support TensorFlow weights
    try:
        model = AutoModel.from_pretrained(MODEL_NAME, from_tf=True)
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

    return model, tokenizer
