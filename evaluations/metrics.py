import sacrebleu
from comet import download_model, load_from_checkpoint
import logging

# Set up logging for errors and info
logging.basicConfig(level=logging.INFO)

import torch
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()


def load_comet_model(model_name="Unbabel/wmt22-comet-da"):
    """Download and load COMET model"""
    try:
        model_path = download_model(model_name)
        model = load_from_checkpoint(model_path)
        logging.info(f"✅ Successfully loaded COMET model: {model_name}")
        return model
    except Exception as e:
        logging.error(f"❌ Failed to load COMET model {model_name}: {str(e)}")
        return None

# Load COMET model
comet_model = load_comet_model("Unbabel/wmt22-comet-da")


def compute_bleu(references, hypotheses, tgt_lang):
    """Compute BLEU score using SacreBLEU with language-specific tokenization."""
    try:
        hypotheses = [hyp if hyp.strip() else "<EMPTY>" for hyp in hypotheses]  

        # Normalize references and hypotheses
        references = [ref.strip() for ref in references]
        hypotheses = [hyp.strip() for hyp in hypotheses]

        # Check for empty hypotheses
        if all(hyp == "<EMPTY>" for hyp in hypotheses):
            logging.warning(f"⚠️ All hypotheses are empty for {tgt_lang}! BLEU score will be low.")
            return 0.0
        
        # Choose tokenizer based on target language
        tokenizer = "flores200" if tgt_lang != "en" else "13a"

        # Compute BLEU with language-specific tokenization
        bleu = sacrebleu.corpus_bleu(hypotheses, [references])
        logging.info(f"📊 BLEU score for {tgt_lang}: {bleu.score}")

        return bleu.score

    except Exception as e:
        logging.error(f"❌ Error computing BLEU score for {tgt_lang}: {str(e)}")
        return None


def compute_comet(references, hypotheses, sources, batch_size=8):
    """Compute COMET score for more accurate evaluation"""
    if not comet_model:
        logging.error("❌ COMET model is not loaded. Cannot compute COMET score.")
        return None

    try:
        # Ensure the sources, hypotheses, and references are aligned correctly
        if len(references) != len(hypotheses) or len(hypotheses) != len(sources):
            logging.error("❌ Mismatched input lengths: references, hypotheses, and sources must be of the same length.")
            return None

        # Format the data for COMET
        data = [{"src": src, "mt": hyp, "ref": ref} for src, hyp, ref in zip(sources, hypotheses, references)]
        comet_scores = comet_model.predict(data, batch_size=batch_size)

        # Log individual scores for debugging
        for idx, score in enumerate(comet_scores.scores):
            logging.info(f"🌍 COMET score for index {idx}: {score}")

        # Return the average score
        avg_comet_score = sum(comet_scores.scores) / len(comet_scores.scores) if comet_scores.scores else None
        logging.info(f"📈 Average COMET score: {avg_comet_score}")
        return avg_comet_score

    except Exception as e:
        logging.error(f"❌ Error computing COMET score: {str(e)}")
        return None
