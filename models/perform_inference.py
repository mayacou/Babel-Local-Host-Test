import torch


def perform_inference(test_data, model_name, model, tokenizer, target_language, config=None):
    """
    Perform inference using a model's generate method for a given test dataset.

    Args:
        test_data (list): List of input texts or dictionaries with a 'source' key.
        model_name (str): The name of the model (for logging or config adjustments).
        model: Loaded model supporting generate() or forward() methods.
        tokenizer: Loaded tokenizer.
        target_language (str): Target language code (e.g., 'fr' for French, 'ja' for Japanese).
        config (dict, optional): Inference parameters (BEAM_SIZE, TEMPERATURE, TOP_K, TOP_P).

    Returns:
        tuple: (generated_translations, reference_translations).
    """
    config = config or {"BEAM_SIZE": 5, "TEMPERATURE": 1.0, "TOP_K": 50, "TOP_P": 0.95}
    generated_translations, reference_translations = [], []

    for example in test_data:
        input_text = example if isinstance(example, str) else example.get('source', '')
        reference_translations.append(input_text.strip())

        try:
            formatted_input = f"<s>[INST] Translate this to {target_language}: {input_text} [/INST]"
            inputs = tokenizer(formatted_input, return_tensors="pt", padding=True, truncation=True).to(model.device)

            with torch.no_grad():
                if hasattr(model, "generate"):
                    output = model.generate(
                        **inputs,
                        num_beams=config["BEAM_SIZE"],
                        temperature=config["TEMPERATURE"],
                        early_stopping=True
                    )
                    generated_translation = tokenizer.decode(output[0], clean_up_tokenization_spaces=True).strip()
                else:
                    logits = model(**inputs).logits
                    predicted_ids = logits.argmax(dim=-1)
                    generated_translation = tokenizer.decode(predicted_ids[0], skip_special_tokens=True)

            generated_translations.append(generated_translation.strip())

        except Exception as e:
            print(f"⚠️ Error during {model_name} inference: {e}")
            generated_translations.append("")

    return generated_translations, reference_translations

