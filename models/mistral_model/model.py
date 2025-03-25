from transformers import AutoModelForCausalLM, AutoTokenizer
from models.mistral_model.config import MODEL_NAME, ATTN_IMPLEMENTATION, DEVICE_MAP, TORCH_DTYPE
import torch
def mistral_load(model_name=MODEL_NAME, 
                 attn_implementation=ATTN_IMPLEMENTATION, 
                 device_map=DEVICE_MAP,
                 torch_dtype=TORCH_DTYPE):
    try:
        # Load the tokenizer
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token  # Set pad_token if it's missing
        torch.cuda.empty_cache()
        torch.cuda.memory_allocated()
        # Load the model with memory optimization settings
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map=device_map,
            attn_implementation=attn_implementation,
            low_cpu_mem_usage=True
        )
        print(torch.cuda.is_available())  # Should return True if CUDA is available
        print(next(model.parameters()).device)  # Should return 'cuda' if running on GPU
        print(torch.cuda.memory_allocated())  # Check used memory
        print(model.hf_device_map)  # Should show most layers on CUDA


        # Return model and tokenizer
        return model, tokenizer
    
    except Exception as e:
        print(f"Error loading model or tokenizer for {model_name}: {e}")
        return None, None
def mistral_inference(test_data, model, tokenizer, src_lang, tgt_lang, config, debug):
    translations = []

    if hasattr(tokenizer, "src_lang"):
        try:
            tokenizer.src_lang = src_lang
        except Exception as e:
            if debug:
                print(f"Error setting tokenizer.src_lang: {e}")

    forced_bos_token_id = None
    if hasattr(tokenizer, "convert_tokens_to_ids"):
        try:
            forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_lang)
        except Exception as e:
            if debug:
                print(f"Error converting tgt_lang '{tgt_lang}' to token ID: {e}")

    for example in test_data:
        input_text = example if isinstance(example, str) else example.get('source', '')

        if not input_text:
            translations.append("")
            continue

        try:
            prompt = f"<s>[INST] Translate this to {tgt_lang}: {input_text}[/INST] <!-- TRANSLATION_START -->"
            inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(model.device)

            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    num_beams=config["BEAM_SIZE"],
                    length_penalty=config["LENGTH_PENALTY"],
                    early_stopping=False,
                    forced_bos_token_id=forced_bos_token_id,
                    max_length=config["MAX_LENGTH"]
                )

            if output is None or output.size(0) == 0:
                translations.append("")
                continue

            decoded_output = tokenizer.batch_decode(output, skip_special_tokens=True)
            if decoded_output:
                translations.append(decoded_output[0].strip())
            else:
                translations.append("")

        except Exception as e:
            if debug:
                print(f"Error during Mistral inference: {e}")
            translations.append("")

    return translations
