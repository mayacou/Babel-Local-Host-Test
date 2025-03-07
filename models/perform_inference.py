# perform_inference.py
import torch

def inference_router(test_data, model_name, model, tokenizer):
    if model_name == "mistral":
        return mistral_inference(test_data, model, tokenizer)
    elif model_name == "madlad":
        return madlad_inference(test_data, model, tokenizer)
    else:
        raise ValueError(f"Model {model_name} is not supported for inference routing.")


from models.madlad_model.config import BEAM_SIZE, TEMPERATURE, TOP_K, TOP_P

def madlad_inference(test_data, model, tokenizer):
    """
    Perform inference using the MADLAD model.

    Args:
        test_data (list): List of input texts or dictionaries containing 'source' keys.
        model: Loaded MADLAD model.
        tokenizer: Loaded tokenizer.
        model_name (str): Name of the model (for logging purposes).

    Returns:
        tuple: Two lists - generated_translations and reference_translations.
    """
    generated_translations = []
    reference_translations = []

    for example in test_data:
        input_text = example if isinstance(example, str) else example.get('source', '')
        reference_translations.append(input_text)  # Add input text to reference translations

        try:
            # Tokenize the input text
            inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True).to(model.device)

            # Generate translation
            with torch.no_grad():
                output = model.generate(  # Use model.generate, not model(inputs)
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    num_beams=BEAM_SIZE,
                    temperature=TEMPERATURE,
                    do_sample=True,
                    top_k=TOP_K,
                    top_p=TOP_P,
                    early_stopping=True,  # Stop generation when the model finishes
                )

            # Decode the generated output
            generated_translation = tokenizer.decode(output[0], skip_special_tokens=True)
            generated_translations.append(generated_translation)

        except Exception as e:
            print(f"⚠️ Error during inference for madlad: {e}")
            generated_translations.append("")  # Append an empty string in case of error

    return generated_translations, reference_translations

# Mistral Inference Logic
def mistral_inference(test_data, model, tokenizer):
    generated_translations = []
    reference_translations = []

    for example in test_data:
        input_text = example if isinstance(example, str) else example.get('source', '')

        # Tokenize input text
        inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True).to(model.device)

        try:
            with torch.no_grad():
                if hasattr(model, "generate"):
                    output = model.generate(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        max_length=100
                    )
                    generated_translation = tokenizer.decode(output[0], skip_special_tokens=True)
                else:
                    output = model.forward(**inputs)
                    logits = output.logits
                    predicted_ids = logits.argmax(dim=-1)
                    generated_translation = tokenizer.decode(predicted_ids[0], skip_special_tokens=True)

            generated_translations.append(generated_translation)
            reference_translations.append(input_text)

        except Exception as e:
            print(f"⚠️ Error during inference for {model_name}: {e}")

    return generated_translations, reference_translations