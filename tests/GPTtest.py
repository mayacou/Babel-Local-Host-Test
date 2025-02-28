from openai import OpenAI 
from dotenv import load_dotenv
import os

# Configuration
RESULTS_JSON = 'gpt_results.json'
BATCH_SIZE = 3
MODEL_NAME = "gpt-4o-mini" 
LANGUAGE_MAP = {
    "sq": "Albanian",
    "bg": "Bulgarian",
    "cs": "Czech",
    "da": "Danish",
    "nl": "Dutch",
    "et": "Estonian",
    "fi": "Finnish",
    "fr": "French",
    "gmq": "Norwegian",  # North Germanic model
    "de": "German",
    "el": "Greek",
    "hu": "Hungarian",
    "it": "Italian",
    "lv": "Latvian",
    "lt": "Lithuanian",
    "mk": "Macedonian",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "sk": "Slovak",
    "es": "Spanish",
    "sv": "Swedish",
    "tr": "Turkish",
    "sla": ["Polish", "Slovenian"],  # Slavic model
}

# Load environment variables from .env file to get OpenAI API key
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Set the OPENAI_API_KEY in the .env file.")

# Initialize the OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY) 

def translate_with_chatgpt(source_sentences, target_language):
    try:
        # Handle special cases in LANGUAGE_MAP
        if target_language in LANGUAGE_MAP and isinstance(LANGUAGE_MAP[target_language], list):
            # Use the first language in the priority list
            target_language = LANGUAGE_MAP[target_language][0]
        else:
            target_language = LANGUAGE_MAP.get(target_language, target_language)

        # Combine all sentences into a single user message
        sentences_str = "\n".join([f"- {sentence}" for sentence in source_sentences])
        user_message = f"Translate the following English sentences to {target_language}:\n{sentences_str}"

        # Prepare the messages for the API call
        messages = [
            {
                "role": "system",
                "content": f"You are a translator that translates English text to {target_language}."
            },
            {
                "role": "user",
                "content": user_message
            }
        ]

        # Call the OpenAI API
        response = client.chat.completions.create(
            model="gpt-4",  # Use a valid model name
            messages=messages,
            max_tokens=1000,
            temperature=0.7
        )

        # Extract the response content
        response_content = response.choices[0].message.content.strip()

        # Split the response into individual translations
        translations = response_content.split("\n")

        # Clean up the translations (remove bullet points or numbering if present)
        translations = [translation.strip("- ").strip() for translation in translations]

        return translations
    except Exception as e:
        print(f"Error during translation: {e}")
        return [""] * len(source_sentences)  # Return empty strings if there's an error