# perform_inference.py

import torch

def perform_inference(test_data, model, tokenizer):
    generated_translations = []
    reference_translations = []

    for example in test_data:
        if isinstance(example, str):
            input_text = example  # If it's a string, just use it directly
        else:
            input_text = example.get('source', '')  # Access using .get() to avoid KeyError
        
        # Tokenize input text
        inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True).to(model.device)

        # Check if the model has the 'generate' method
        if hasattr(model, 'generate'):
            # Use the generate method (if supported by the model)
            output = model.generate(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], max_length=100)
            generated_translation = tokenizer.decode(output[0], skip_special_tokens=True)
        else:
            with torch.no_grad():
                output = model(**inputs)  
            
            logits = output.logits  # Assuming the model returns logits (adjust as needed)
            predicted_ids = logits.argmax(dim=-1)  # Get the most likely token IDs from the logits
            generated_translation = tokenizer.decode(predicted_ids[0], skip_special_tokens=True)

        # Store the generated and reference translations
        generated_translations.append(generated_translation)
        reference_translations.append(input_text)

    return generated_translations, reference_translations
