import json

def load_data_from_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Extract inputs and references into separate lists
    source_sentences = [item["input"] for item in data]
    reference_sentences = [item["reference"] for item in data]
    return source_sentences, reference_sentences