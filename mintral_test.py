import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
)

# Load the Mistral model and tokenizer
model_name = "mistralai/Mistral-7B-v0.1"  
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1", 
                                             quantization_config=quantization_config,
                                             attn_implementation="flash_attention_2", 
                                             device_map="auto")

# Sample input
input_text = "Translate the following English sentence into French: 'Hello, how are you?'"
inputs = tokenizer(input_text, return_tensors="pt").to("cuda")

# Generate response
output = model.generate(**inputs, max_new_tokens=100)
response = tokenizer.decode(output[0], skip_special_tokens=True)

print("Mistral Output:", response)
