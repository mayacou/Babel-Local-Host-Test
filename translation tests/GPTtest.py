from openai import OpenAI 
from evaluation import evaluate_bleu_and_comet
from dotenv import load_dotenv
from load_data import load_data_from_json
import os

# Configuration
DATA_JSON = "data.json"
BATCH_SIZE = 1
MODEL_NAME = "gpt-4o-mini" 

# Load environment variables from .env file
load_dotenv()

# Get the OpenAI API key from the environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Please set the OPENAI_API_KEY in the .env file.")

# Initialize the OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY) 

# Translate a batch of sentences using the OpenAI API
def translate_with_chatgpt(batch, target_language):
    try:
        # Prepare the messages for the API call
        messages = [
            {
                "role": "system",
                "content": f"You are a translator that translates English text to {target_language}."
            }
        ]
        for sentence in batch:
            messages.append({"role": "user", "content": f"Translate the following English text to {target_language}: {sentence}"})

        # Call the OpenAI API
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=1000,  
            temperature=0.7,  
        )

        # Extract translations from the response
        translations = [
            choice.message.content.strip() for choice in response.choices
        ]
        return translations
    except Exception as e:
        print(f"Error during translation: {e}")
        return [""] * len(batch)  # Return empty strings if there's an error

def main():
    # Load source and reference from data.json
    source_sentences, reference_sentences = load_data_from_json(DATA_JSON)

    # Translate 
    translated_sentences = []
    for i in range(0, len(source_sentences), BATCH_SIZE):
        batch = source_sentences[i : i + BATCH_SIZE]
        batch_translations = translate_with_chatgpt(batch, target_language="French")
        translated_sentences.extend(batch_translations)
        print(f"Batch {i//BATCH_SIZE + 1} done.")
    print("Done translating.")
    
    # Print all translations
    #print("\nTranslations:")
    #for idx, (source, translation) in enumerate(zip(source_sentences, translated_sentences)):
    #    print(f"{idx + 1}. Source: {source}")
    #    print(f"   Translation: {translation}\n")

    # Evaluate translations
    evaluate_bleu_and_comet(
        source_sentences=source_sentences,
        translated_sentences=translated_sentences,
        reference_sentences=reference_sentences
    )

if __name__ == "__main__":
    main()