def perform_inference(test_data, model, tokenizer):
    generated_translations = []
    reference_translations = []

    for example in test_data:
        if isinstance(example, str):
            input_text = example  # If it's a string, just use it directly
        else:
            input_text = example.get('source', '')  # Access using .get() to avoid KeyError
        
        # Tokenize input text
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        
        # Generate translation
        output = model.generate(**inputs, max_new_tokens=100)
        generated_translation = tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Store the generated and reference translations
        generated_translations.append(generated_translation)
        reference_translations.append(input_text)  # Append the input text as the reference

    return generated_translations, reference_translations
