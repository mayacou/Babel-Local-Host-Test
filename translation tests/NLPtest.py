from transformers import pipeline
from evaluation import evaluate_bleu_and_comet
from load_data import load_data_from_json

DATA_JSON = "data.json"
# adjust based on sample size
BATCH_SIZE = 10
# The models should only be helsinki, other hugging face models will need a system prompt to tell it how to translate.
MODEL_NAME = "Helsinki-NLP/opus-mt-en-fr"

def main():
    # Load source and reference from data.json
    source_sentences, reference_sentences = load_data_from_json(DATA_JSON)

    # Translate using Hugging Face model
    translated_sentences = []
    
    # Load translation model (Example: English to French)
    translator = pipeline("translation", model=MODEL_NAME)
    # translates in batches to reduce model calls
    for i in range(0, len(source_sentences), BATCH_SIZE):
        batch = source_sentences[i : i + BATCH_SIZE]
        outputs = translator(batch)
        # outputs is a list of dicts like [{"translation_text": ...}, ...]
        batch_translations = [out["translation_text"] for out in outputs]
        # add to translated sentences
        translated_sentences.extend(batch_translations)
        print(f"Batch {i//BATCH_SIZE + 1} done.")
    print("Done translating.")
    
    evaluate_bleu_and_comet(
        source_sentences=source_sentences,
        translated_sentences=translated_sentences,
        reference_sentences=reference_sentences
    )

if __name__ == "__main__":
    main()
