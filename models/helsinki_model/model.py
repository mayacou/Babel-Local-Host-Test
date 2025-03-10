from transformers import MarianMTModel, MarianTokenizer

def helsinki_load(model_name):
   tokenizer = MarianTokenizer.from_pretrained(model_name)
   model = MarianMTModel.from_pretrained(model_name)
   return model, tokenizer