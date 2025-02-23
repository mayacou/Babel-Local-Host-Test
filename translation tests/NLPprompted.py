from transformers import pipeline
from evaluation import evaluate_bleu_and_comet
from load_data import load_data_from_json

# Global configuration
DATA_JSON = "data.json"
BATCH_SIZE = 10

# Model-specific configuration
MODEL_NAME = "facebook/bart-base" 
PIPELINE_TYPE = "text2text-generation" 
PROMPT_TEMPLATE = "translate English to French: {sentence}" 

# Translate a batch of sentences using the specified model
def translate_batch(model, batch, target_language="French"):
   try:
      # Prepare the prompts for translation
      prompts = [
         PROMPT_TEMPLATE.format(sentence=sentence)
         for sentence in batch
      ]
        
      # Generate translations using the model
      outputs = model(prompts, max_length=100, num_return_sequences=1, truncation=True)
        
      # Extract translations from the outputs
      if PIPELINE_TYPE == "text2text-generation":
         translations = [out["generated_text"].strip() for out in outputs]
      elif PIPELINE_TYPE == "text-generation":
         translations = [out[0]["generated_text"].strip() for out in outputs]
      else:
         raise ValueError(f"Unsupported pipeline type: {PIPELINE_TYPE}")
        
      return translations
   except Exception as e:
      print(f"Error during translation: {e}")
      return [""] * len(batch)  # Return empty strings if there's an error

def main():
   # Load source and reference from data.json
   source_sentences, reference_sentences = load_data_from_json(DATA_JSON)

   # Load the model with the specified pipeline type
   model = pipeline(PIPELINE_TYPE, model=MODEL_NAME)

   # Translate using the model
   translated_sentences = []
   for i in range(0, len(source_sentences), BATCH_SIZE):
      batch = source_sentences[i : i + BATCH_SIZE]
      batch_translations = translate_batch(model, batch, target_language="French")
      translated_sentences.extend(batch_translations)
      print(f"Batch {i//BATCH_SIZE + 1} done.")
   print("Done translating.")

   # Print all translations
   print("\nTranslations:")
   for idx, (source, translation) in enumerate(zip(source_sentences, translated_sentences)):
      print(f"{idx + 1}. Source: {source}")
      print(f"   Translation: {translation}\n")

   # Evaluate translations
   evaluate_bleu_and_comet(
      source_sentences=source_sentences,
      translated_sentences=translated_sentences,
      reference_sentences=reference_sentences
   )

if __name__ == "__main__":
   main()