import csv
from openai import OpenAI
from dotenv import load_dotenv
import os
from helpers.evaluation import compute_bleu, compute_comet
from datasets_loader.load_wmt import load_wmt_data
from datasets_loader.load_tedTalk import load_tedTalk_data
from datasets_loader.load_europarl import load_europarl_data

# Load OpenAI API key from environment
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

# Ensure directories exist
os.makedirs("data", exist_ok=True)
os.makedirs("translation_results", exist_ok=True)

def translate(source_sentences, target_language, reference_sentences):
    """Translate a batch of sentences using ChatGPT API and compute evaluation scores."""
    try:
        # Combine sentences into a single user message
        sentences_str = "\n".join([f"- {sentence}" for sentence in source_sentences])
        user_message = f"Translate the following English sentences to {target_language}:\n{sentences_str}"

        messages = [
            {"role": "system", "content": f"You are a translator that translates English text to {target_language}."},
            {"role": "user", "content": user_message}
        ]

        # Call OpenAI API
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=3000,
            temperature=0.7
        )

        # Extract response content
        response_content = response.choices[0].message.content.strip()
        translations = response_content.split("\n")
        translations = [t.strip("- ").strip() for t in translations]

        # Compute BLEU and COMET scores
        bleu = compute_bleu(reference_sentences, translations)
        comet = compute_comet(reference_sentences, translations, source_sentences)

        return translations, bleu, comet
    except Exception as e:
        print(f"❌ Translation error: {e}")
        return ["ERROR"] * len(source_sentences), "NA", "NA"  # Return error placeholders

# ✅ Open CSV files only when needed
for dataset_name, dataset_loader in DATASETS.items():
    try:
        language_pairs = dataset_loader("get_languages")

        for language in language_pairs:
            try:
                source_sentences, reference_sentences = dataset_loader(language)

                # ✅ Skip if no data found
                if not source_sentences or not reference_sentences:
                    print(f"⚠️ No dataset for {language}. Skipping.")
                    with open(RESULTS_CSV, mode="a", newline="") as file:
                        writer = csv.writer(file)
                        writer.writerow([dataset_name, language, "NA", "NA"])
                    continue

                translations, bleu, comet = translate(source_sentences, language, reference_sentences)

                # ✅ Write results to CSV
                with open(RESULTS_CSV, mode="a", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow([dataset_name, language, round(bleu, 2), round(comet, 2)])

                # ✅ Write translations to CSV
                with open(TRANSLATIONS_CSV, mode="a", newline="") as trans_file:
                    trans_writer = csv.writer(trans_file)
                    for src, translation, ref in zip(source_sentences, translations, reference_sentences):
                        trans_writer.writerow([dataset_name, language, src, translation, ref])

                print(f"✅ {dataset_name} | {language} -> BLEU: {bleu}, COMET: {comet}")

            except Exception as e:
                print(f"⚠️ Skipping {dataset_name} for {language}: {e}")
                with open(RESULTS_CSV, mode="a", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow([dataset_name, language, "Error", "Error"])

    except Exception as e:
        print(f"❌ Failed to process dataset {dataset_name}: {e}")
