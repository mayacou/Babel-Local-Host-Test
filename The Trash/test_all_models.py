import pytest
from helpers.model_loader import load_model, translate_text

# List of models to test
MODELS_TO_TEST = [
    ("Helsinki-NLP/opus-mt-en-fr", "Hello", "Bonjour"),  # English → French
    ("Helsinki-NLP/opus-mt-en-es", "Hello", "Hola"),  # English → Spanish
    ("Helsinki-NLP/opus-mt-en-de", "Hello", "Hallo"),  # English → German
]

@pytest.mark.parametrize("model_name, input_text, expected_output", MODELS_TO_TEST)
def test_models(model_name, input_text, expected_output):
    """Test multiple Hugging Face translation models."""
    model, tokenizer = load_model(model_name)
    translated_text = translate_text(model, tokenizer, input_text)
    assert expected_output in translated_text, f"Expected '{expected_output}', but got '{translated_text}'"
