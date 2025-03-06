# models/mistral/mistral_model.py

from transformers import AutoModelForCausalLM, AutoTokenizer
from models.mistral_model.config import MODEL_NAME, QUANTIZATION_CONFIG, ATTN_IMPLEMENTATION, DEVICE_MAP, TORCH_DTYPE

def mistral_load(model_name=MODEL_NAME, 
                             quantization_config=QUANTIZATION_CONFIG, 
                             attn_implementation=ATTN_IMPLEMENTATION, 
                             device_map=DEVICE_MAP,
                             torch_dtype=TORCH_DTYPE
                             ):
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Load the model with the configuration
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device_map,
        # quantization_config=quantization_config,
        attn_implementation=attn_implementation
    )
    
    return model, tokenizer


    generated_translations = []
    reference_translations = []

    for example in test_data:
        # print("Example:", example)  # Print the current example to inspect its structure
        # print(type(example))  # Print the type of the current example

        # If 'example' is a string, you can't access it like a dictionary
        if isinstance(example, str):
            input_text = example  # If it's a string, just use it directly
        else:
            # If it's a dictionary, ensure it has the 'source' key
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