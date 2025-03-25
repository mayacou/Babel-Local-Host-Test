from models.mistral_model.model import mistral_inference  
from models.bert_model.model import bert_inference
def clean_output(text):
    start_marker = "<!-- TRANSLATION_START -->"
    end_marker = "<!-- TRANSLATION_END -->"
    
    if start_marker in text:
        text = text.split(start_marker)[-1]  # Only take the part after the start marker
    
    if end_marker in text:
        text = text.split(end_marker)[0]  # Only take the part before the end marker
    
    return text.strip()

def perform_inference(test_data, model, tokenizer, src_lang, tgt_lang, model_name, config=None, debug=True):
    config = config or {"BEAM_SIZE": 5, "LENGTH_PENALTY": 1.2, "MAX_LENGTH": 400}
    generated_translations = []

    match model_name:
        case "mistral":
            generated_translations = mistral_inference(test_data, model, tokenizer, src_lang, tgt_lang, config, debug)
        case "bert":
            generated_translations = bert_inference(test_data, model, tokenizer, src_lang, tgt_lang, config, debug)
        case _:
            raise ValueError(f"Inference for model {model_name} is not implemented.")

    # Clean all translations before returning
    return [clean_output(text) for text in generated_translations]
