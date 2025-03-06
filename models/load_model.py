# load_model_and_tokenizer.py

from models.mistral_model.model import mistral_load  

def load_model_and_tokenizer(model_name="mistral"):
    if model_name == "mistral":
        return mistral_load()
    else:
        raise ValueError(f"Model {model_name} is not supported in this hub.")
