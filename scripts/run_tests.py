# scripts/run_tests.py

from config.languages import LANGUAGES
from scripts.data_loader import load_opus_data
from models.mistral_model.model import load_model_and_tokenizer, perform_inference
from evaluations.metrics import compute_bleu, compute_comet

def run_test_for_language_pair(language_pair):
    first_language_name = LANGUAGES.get(language_pair[0], "Unknown Language")
    second_language_name = LANGUAGES.get(language_pair[1], "Unknown Language")

    print(f"Running test for {first_language_name} -> {second_language_name}...")

    # Load the data
    test_data = load_opus_data(language_pair)
    if not test_data:
        print(f"Error: Data for {first_language_name} -> {second_language_name} not found!")
        return -1, -1
    
    # Extract source sentences (original sentences) from the test data
    source_sentences = [example['translation'][language_pair[0]] for example in test_data['test']]
    print(f"Source Sentences for {first_language_name} -> {second_language_name}: {source_sentences}")

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer()

    # Perform inference
    generated_translations, reference_translations = perform_inference(test_data, model, tokenizer)
    print(f"Generated Translations: {generated_translations}")
    print(f"Reference Translations: {reference_translations}")
    
    # Evaluate translations using BLEU and COMET
    bleu_score = compute_bleu(generated_translations, reference_translations)
    comet_score = compute_comet(generated_translations, reference_translations, source_sentences)  
    
    print(f"BLEU Score: {bleu_score}")
    print(f"COMET Score: {comet_score}")
    
    return bleu_score, comet_score
