import csv
from openai import OpenAI 
from dotenv import load_dotenv
import os
from helpers.evaluation import compute_bleu, compute_comet
from datasets_loader.load_wmt import load_wmt_data
from datasets_loader.load_tedTalk import load_tedTalk_data
from datasets_loader.load_europarl import load_europarl_data

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("Set the OPENAI_API_KEY in the .env file.")
client = OpenAI(api_key=openai_api_key)

# Datasets to test
DATASETS = {
    "WMT": load_wmt_data,
    "TED": load_tedTalk_data,
    "Europarl": load_europarl_data
}

# Configuration
RESULTS_CSV = "data/ChatGPT_test_results.csv"
TRANSLATIONS_CSV = "translation_results/ChatGPT_translations.csv"
MODEL_NAME = "gpt-4o-mini" 

def translate(source_sentences, target_language, reference_sentences):
    try:
        # Combine all sentences into a single user message
        sentences_str = "\n".join([f"- {sentence}" for sentence in source_sentences])
        user_message = f"Translate the following English sentences to {target_language}:\n{sentences_str}"

        # Prepare the messages for the API call
        messages = [
            {"role": "system", "content": f"You are a translator that translates English text to the language with the language code {target_language}."},
            {"role": "user", "content": user_message}
        ]

        # Call the OpenAI API
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=1000,
            temperature=0.7
        )

        # Extract the response content
        response_content = response.choices[0].message.content.strip()
        translations = response_content.split("\n")
        translations = [translation.strip("- ").strip() for translation in translations]

        bleu = compute_bleu(reference_sentences, translations)
        comet = compute_comet(reference_sentences, translations, source_sentences)
        
        return translations, bleu, comet
    except Exception as e:
        print(f"Error during translation: {e}")
        return ["Error"] * len(source_sentences), 0, 0  # Return error placeholders

with open(RESULTS_CSV, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Dataset", "Language", "BLEU", "COMET"])
    
    with open(TRANSLATIONS_CSV, mode="w", newline="") as trans_file:
        trans_writer = csv.writer(trans_file)
        trans_writer.writerow(["Dataset", "Language", "Source Sentence", "Translation", "Reference Sentence"])
        
        for dataset_name, dataset_loader in DATASETS.items():
            try:
                language_pairs = dataset_loader("get_languages")
                for language in language_pairs:
                    source_sentences, reference_sentences = dataset_loader(language)
                    translations, bleu, comet = translate(source_sentences, language, reference_sentences)
                    
                    writer.writerow([dataset_name, language, round(bleu, 2), round(comet, 2)])
                    file.flush()
                    
                    for source, translation, reference in zip(source_sentences, translations, reference_sentences):
                        trans_writer.writerow([dataset_name, language, source, translation, reference])
                    trans_file.flush()
            except Exception as e:
                print(f"⚠️ Skipping {dataset_name} for {language}: {e}")
