from transformers import MBartForConditionalGeneration, MBartTokenizer

def bert_load(model_name="facebook/mbart-large-50-many-to-many-mmt"):
    # Load the mBART tokenizer and model
    tokenizer = MBartTokenizer.from_pretrained(model_name)
    model = MBartForConditionalGeneration.from_pretrained(model_name)

    return model, tokenizer
