from transformers import AutoTokenizer, BertConfig, BertModel
from models.bert_model.config import MODEL_NAME, TOKENIZER_NAME, DEVICE_MAP, TORCH_DTYPE, PAD_TOKEN

def bert_load():
    """
    Load a multilingual BERT model and tokenizer for English-to-Multi-Language tasks.

    Returns:
        model: Loaded BERT model (potentially for fine-tuning).
        tokenizer: Loaded tokenizer for multilingual BERT.
    """
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    # Ensure the tokenizer has a padding token
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': PAD_TOKEN})

    # Initializing a multilingual BERT configuration
    configuration = BertConfig.from_pretrained(MODEL_NAME)

    # Load the BERT model with the specified configuration
    model = BertModel.from_pretrained(MODEL_NAME, config=configuration)

    return model, tokenizer
