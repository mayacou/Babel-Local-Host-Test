import os
import subprocess
import shutil

CACHE_DIR = os.path.expanduser("~/.cache/huggingface")  # Default cache location for Hugging Face models

# Function to clear the Hugging Face cache
def clear_huggingface_cache():
    if os.path.exists(CACHE_DIR):
        print(f"Clearing Hugging Face cache at {CACHE_DIR}...")
        shutil.rmtree(CACHE_DIR)
    else:
        print("No Hugging Face cache found to clear.")

# Function to test a model
def run_model_test(model_name):
    print(f"Running model test for {model_name}...")

    # Clear cache before loading the model
    clear_huggingface_cache()

    # Run the model test script with the model name
    try:
        subprocess.run(["python3", "model_test.py", model_name], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running model test for {model_name}: {e}")
        
# Main function to run the models
def main():
    models = ["mistral", "madlad", "m2m", "nllb", "tower instruct"]

    for model in models:
        run_model_test(model)

if __name__ == "__main__":
    main()
