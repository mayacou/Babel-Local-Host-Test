import pytest
import re
from helpers.model_loader import load_model, translate_text

MODEL_NAME = "Helsinki-NLP/opus-mt-en-fr"  # English to French

@pytest.fixture
def setup_model():
    """Load the model before running tests."""
    return load_model(MODEL_NAME)

def test_translation_accuracy(setup_model):
    model, tokenizer = setup_model
    input_text = "Hello, how are you?"
    expected_pattern = r"Bonjour, comment allez-vous\s?\?"

    translated_text = translate_text(model, tokenizer, input_text)

    assert re.match(expected_pattern, translated_text), (
        f"Expected a match for pattern '{expected_pattern}', but got '{translated_text}'"
    )


def test_translation_non_empty(setup_model):
    """Ensure translation output is not empty."""
    model, tokenizer = setup_model
    translated_text = translate_text(model, tokenizer, "This is a test.")
    assert translated_text.strip() != "", "Translation output is empty!"
