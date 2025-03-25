import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import traceback

def clean_output(text):
    # Clean the output by removing everything before and after the markers
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

    tokenizer.src_lang = src_lang
    print("CHECKING SOMETHING")
    forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_lang)
    print("CHECKING SOMETHING 2")

    for example in test_data:
        input_text = example if isinstance(example, str) else example.get('source', '')

        if not input_text:
            generated_translations_src_to_tgt.append("")
            continue

        try:
            # Construct the prompt for translation
            prompt = f"<s>[INST] Translate this to {tgt_lang}: {input_text}[/INST] <!-- TRANSLATION_START -->"
            
            if debug:
                print(f"Prompt: {prompt}")  # Debugging: print the prompt

            inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(model.device)
            
            if debug:
                print(f"Inputs: {inputs}")  # Debugging: print tokenized inputs

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

            # Decode tensor output safely
            decoded_output = tokenizer.batch_decode(output, skip_special_tokens=True)

            if debug:
                print(f"Decoded output: {decoded_output}")  # Debugging: check decoded output

            if decoded_output:
                # Clean the output to remove anything before or after the marker
                generated_translation_src_to_tgt = clean_output(decoded_output[0])
                generated_translations_src_to_tgt.append(generated_translation_src_to_tgt.strip())
            else:
                generated_translations_src_to_tgt.append("")
                continue

        except Exception as e:
            if debug:
                print(f"Error: {e}")  # Debugging: print error if one occurs
                traceback.print_exc()  # Print full traceback for debugging
            generated_translations_src_to_tgt.append("")

    return generated_translations_src_to_tgt
