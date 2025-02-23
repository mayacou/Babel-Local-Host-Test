import sacrebleu
from comet import download_model, load_from_checkpoint

def evaluate_bleu_and_comet(source_sentences, translated_sentences, reference_sentences):
   """
   Computes and prints the BLEU score using sacrebleu and the 
   COMET score using the 'Unbabel/wmt22-comet-da' model.
   """

   # --- BLEU Evaluation ---
   # sacrebleu expects references as a list of lists: [[ref1, ref2, ...], [ref1, ref2, ...], ...]
   reference_translations = [[ref] for ref in reference_sentences]
   bleu = sacrebleu.corpus_bleu(translated_sentences, reference_translations)
   print(f"BLEU Score: {bleu.score:.2f}")

   # --- COMET Evaluation ---
   # Download/load COMET model
   comet_model_path = download_model("Unbabel/wmt22-comet-da")
   comet_model = load_from_checkpoint(comet_model_path)

   # Prepare data for COMET
   comet_data = [
      {"src": src, "mt": mt, "ref": ref}
      for src, mt, ref in zip(source_sentences, translated_sentences, reference_sentences)
   ]

   # Predict with COMET
   comet_result = comet_model.predict(comet_data)
   scores = comet_result["scores"]
   avg_score = sum(scores) / len(scores)
   print(f"Average COMET Score: {avg_score:.4f}")
