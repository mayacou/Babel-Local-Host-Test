import google.generativeai as genai
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

def load_gemini():
    """
    Load the Gemini model using the API key.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("❌ Missing Gemini API key. Please set it in the .env file or environment variables.")
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-pro-002")  # Adjust model name if needed
        print("✅ Successfully loaded Gemini model!")
        return model
    except Exception as e:
        print(f"❌ Error loading Gemini: {e}")
        return None

def translate_text(model, text):
    """
    Translate text using the Gemini model.
    """
    if model is None:
        print("❌ Model is not loaded.")
        return ""
    
    try:
        response = model.generate_content(text)
        
        # Ensure the response has valid text
        if response and hasattr(response, "text") and response.text:
            return response.text.strip()
        else:
            print("⚠️ Empty or unexpected response from Gemini.")
            return ""
    
    except Exception as e:
        print(f"❌ Translation error: {e}")
        return ""

