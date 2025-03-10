import torch

def perform_inference(test_data, model_name, model, tokenizer, config=None):
    """
    Perform inference using a model's generate method for any given test data.
    
    Args:
        test_data (list): List of input texts or dictionaries containing a 'source' key.
        model_name (str): The name of the model (used for logging or config adjustments).
        model: Loaded model that supports generate() or forward() methods.
        tokenizer: Loaded tokenizer.
        config (dict, optional): Configuration for inference parameters (e.g., BEAM_SIZE, TEMPERATURE, TOP_K, TOP_P).
        
    Returns:
        tuple: Two lists - generated_translations and reference_translations.
    """
    # Default config if none is provided
    if config is None:
        config = {
            "BEAM_SIZE": 5,
            "TEMPERATURE": 1.0,
            "TOP_K": 50,
            "TOP_P": 0.95
        }

    generated_translations = []
    reference_translations = []

    for example in test_data:
        input_text = example if isinstance(example, str) else example.get('source', '')
        reference_translations.append(input_text)  # Store the original text as a reference

        try:
            # Tokenize the input text
            inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True).to(model.device)

            with torch.no_grad():
                # Check if the model has a generate method (for most seq2seq models)
                if hasattr(model, "generate"):
                    output = model.generate(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        num_beams=config.get("BEAM_SIZE", 5),  # Set beam size for beam search
                        temperature=config.get("TEMPERATURE", 1.0),  # Sampling temperature
                        do_sample=True,  # Whether to sample or use greedy decoding
                        top_k=config.get("TOP_K", 50),  # Top-k filtering for sampling
                        top_p=config.get("TOP_P", 0.95),  # Top-p (nucleus) sampling
                        early_stopping=True  # Stop generation when the model deems it finished
                    )
                    generated_translation = tokenizer.decode(output[0], skip_special_tokens=True)
                else:
                    # Fallback to the forward pass if generate() is unavailable
                    output = model.forward(**inputs)
                    logits = output.logits
                    predicted_ids = logits.argmax(dim=-1)
                    generated_translation = tokenizer.decode(predicted_ids[0], skip_special_tokens=True)

            generated_translations.append(generated_translation)

        except Exception as e:
            print(f"⚠️ Error during {model_name} inference: {e}")
            generated_translations.append("")  # Append an empty string in case of error

    return generated_translations, reference_translations
