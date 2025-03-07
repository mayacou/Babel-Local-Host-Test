from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from models.madlad_model.config import MODEL_NAME, TOKENIZER_NAME, DEVICE_MAP, TORCH_DTYPE, PAD_TOKEN

def madlad_load():
    """
    Load the MADLAD model and tokenizer.

    Returns:
        model: Loaded MADLAD model.
        tokenizer: Loaded tokenizer.
    """
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    # Ensure the tokenizer has a padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = PAD_TOKEN if PAD_TOKEN else tokenizer.eos_token

    # Load the model
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=TORCH_DTYPE,
        device_map=DEVICE_MAP
    )

    return model, tokenizer