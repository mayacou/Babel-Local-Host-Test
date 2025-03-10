import torch

def perform_inference(test_data, model_name, model, tokenizer, target_language, config=None):
    """
    Perform inference using a model's generate method for any given test data.

    Args:
        test_data (list): List of input texts or dictionaries containing a 'source' key.
        model_name (str): The name of the model (used for logging or config adjustments).
        model: Loaded model that supports generate() or forward() methods.
        tokenizer: Loaded tokenizer.
        target_language (str): Target language code (e.g., 'fr' for French, 'ja' for Japanese).
        config (dict, optional): Configuration for inference parameters (e.g., BEAM_SIZE, TEMPERATURE, TOP_K, TOP_P).

    Returns:
        tuple: Two lists - generated_translations and reference_translations.
    """
    # Default configuration
    if config is None:
        config = {
            "BEAM_SIZE": 5,
            "TEMPERATURE": 1.0,
            "TOP_K": 50,
            "TOP_P": 0.95
        }

    generated_translations = []
    reference_translations = []

    # Iterate through each example in the test data
    for example in test_data:
        # Extract input text from dictionary if it's provided, else use the string directly
        input_text = example if isinstance(example, str) else example.get('source', '')
        reference_translations.append(input_text.strip())  # Store the original text as a reference

        try:
            # Format input text for translation
            formatted_input = f"translate to {target_language}: {input_text}"

            # Tokenize the input text
            inputs = tokenizer(formatted_input, return_tensors="pt", padding=True, truncation=True).to(model.device)

            with torch.no_grad():
                if hasattr(model, "generate"):
                    # Use the generate method for beam search without sampling
                    output = model.generate(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        num_beams=config.get("BEAM_SIZE", 5),
                        temperature=config.get("TEMPERATURE", 1.0),
                        do_sample=False,  # Disable sampling for deterministic output
                        top_k=config.get("TOP_K", 50),
                        top_p=config.get("TOP_P", 0.95),
                        early_stopping=True
                    )
                    generated_translation = tokenizer.decode(output[0], skip_special_tokens=True)
                else:
                    # If the model doesn't have generate, use forward method
                    output = model.forward(**inputs)
                    logits = output.logits
                    predicted_ids = logits.argmax(dim=-1)
                    generated_translation = tokenizer.decode(predicted_ids[0], skip_special_tokens=True)

            generated_translations.append(generated_translation.strip())

        except Exception as e:
            print(f"⚠️ Error during {model_name} inference: {e}")
            generated_translations.append("")  # Append empty string in case of error

    return generated_translations, reference_translations
