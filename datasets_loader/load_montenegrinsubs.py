from datasets import load_dataset

BATCH_SIZE = 20

def load_montenegrin_data(target_language):
   try:
      if target_language != "cnr": return [], []
      
      print(f"🟢 Loading dataset: Montenegrin\n")
      dataset = load_dataset("Helsinki-NLP/opus_montenegrinsubs", split="train", trust_remote_code=True).shuffle(seed=42)

      source_sentences = []
      reference_sentences = []

      count = 0
      i = 0
      while count < BATCH_SIZE:
         item = dataset[i]
         i+=1

         if "translation" in item and "en" in item["translation"] and "me" in item["translation"]:
            source = item["translation"]["en"]
            if len(source.split()) > 10:
               source_sentences.append(source)
               reference_sentences.append(item["translation"]["me"])
               count+=1

      return source_sentences, reference_sentences

   except Exception as e:
      print(f"❌ Error loading dataset for {target_language}: {e}")
      return [], []

