import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load .env variables
load_dotenv()

# Configure API key
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# List available models
models = genai.list_models()
for model in models:
    print(model.name)
