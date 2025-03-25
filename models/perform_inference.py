import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import traceback

def clean_output(text):
    start_marker = "<!-- TRANSLATION_START -->"
    end_marker = "<!-- TRANSLATION_END -->"
    
    if start_marker in text:
        text = text.split(start_marker)[-1]  # Only take the part after the start marker
    
    if end_marker in text:
        text = text.split(end_marker)[0]  # Only take the part before the end marker
    
    return text.strip()

def perform_inference(test_data, model, tokenizer, src_lang, tgt_lang, config=None, debug=True):
    config = config or {"BEAM_SIZE": 5, "LENGTH_PENALTY": 1.2, "MAX_LENGTH": 400}
    generated_translations_src_to_tgt = []

    # Safe check for setting src_lang
    valid_langs = getattr(tokenizer, "langs", None)
    print(valid_langs)
    try:
        if hasattr(tokenizer, 'src_lang'):
            print(f"Setting tokenizer.src_lang to {src_lang}")
            tokenizer.src_lang = "english"
            print(f"Successfully set tokenizer.src_lang to {tokenizer.src_lang}")
        else:
            print(f"Tokenizer does not support 'src_lang'. Skipping this step.")
            tokenizer.src_lang = None  # Ensure no value is assigned
    except Exception as e:
        if debug:
            print(f"Error setting tokenizer.src_lang: {e}")
        tokenizer.src_lang = None  # Set to None or a safe default
        print(f"Using default tokenizer.src_lang: {tokenizer.src_lang}")

    # Safe check for converting tgt_lang to token id
    try:
        if hasattr(tokenizer, 'convert_tokens_to_ids'):
            forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_lang)
            print(f"Successfully converted {tgt_lang} to token ID: {forced_bos_token_id}")
        else:
            print(f"Tokenizer does not support 'convert_tokens_to_ids'. Skipping this step.")
            forced_bos_token_id = None  # Handle fallback or default behavior
            print(f"Using default forced_bos_token_id: {forced_bos_token_id}")
    except Exception as e:
        if debug:
            print(f"Error converting {tgt_lang} to token ID: {e}")
        forced_bos_token_id = None  # Handle fallback or default behavior
        print(f"Using default forced_bos_token_id: {forced_bos_token_id}")

    for example in test_data:
        input_text = example if isinstance(example, str) else example.get('source', '')

        if not input_text:
            generated_translations_src_to_tgt.append("")
            continue

        try:
            prompt = f"<s>[INST] Translate this to {tgt_lang}: {input_text}[/INST] <!-- TRANSLATION_START -->"
            
            if debug:
                print(f"Prompt: {prompt}")

            inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(model.device)
            
            if debug:
                print(f"Inputs: {inputs}")

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
                generated_translations_src_to_tgt.append("")
                continue

            decoded_output = tokenizer.batch_decode(output, skip_special_tokens=True)

            if debug:
                print(f"Decoded output: {decoded_output}")

            if decoded_output:
                generated_translation_src_to_tgt = clean_output(decoded_output[0])
                generated_translations_src_to_tgt.append(generated_translation_src_to_tgt.strip())
            else:
                generated_translations_src_to_tgt.append("")
                continue

        except Exception as e:
            if debug:
                print(f"Error during inference: {e}")
                traceback.print_exc()
            generated_translations_src_to_tgt.append("")

    return generated_translations_src_to_tgt
