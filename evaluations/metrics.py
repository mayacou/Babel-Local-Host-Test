import sacrebleu
from comet import download_model, load_from_checkpoint
import logging

# Set up logging for errors and info
logging.basicConfig(level=logging.INFO)

def load_comet_model(model_name="Unbabel/wmt22-comet-da"):
    """Download and load COMET model"""
    try:
        model_path = download_model(model_name)
        model = load_from_checkpoint(model_path)
        logging.info(f"Successfully loaded COMET model: {model_name}")
        return model
    except Exception as e:
        logging.error(f"Failed to load COMET model {model_name}: {str(e)}")
        return None

# Load COMET model
comet_model = load_comet_model("Unbabel/wmt22-comet-da")

def compute_bleu(references, hypotheses):
    """Compute BLEU score using SacreBLEU"""
    try:
        bleu = sacrebleu.corpus_bleu(hypotheses, [references])
        logging.info(f"BLEU score: {bleu.score}")
        return bleu.score
    except Exception as e:
        logging.error(f"Error computing BLEU score: {str(e)}")
        return None

def compute_comet(references, hypotheses, sources, batch_size=8):
    """Compute COMET score for more accurate evaluation"""
    if not comet_model:
        logging.error("COMET model is not loaded. Cannot compute COMET score.")
        return None

    try:
        # Ensure the sources, hypotheses, and references are aligned correctly
        if len(references) != len(hypotheses) or len(hypotheses) != len(sources):
            logging.error("Mismatched input lengths: references, hypotheses, and sources must be of the same length.")
            return None

        # Format the data for COMET
        data = [{"src": src, "mt": hyp, "ref": ref} for src, hyp, ref in zip(sources, hypotheses, references)]
        comet_scores = comet_model.predict(data, batch_size=batch_size)

        # Return the average score
        avg_comet_score = sum(comet_scores.scores) / len(comet_scores.scores) if comet_scores.scores else None
        logging.info(f"COMET score: {avg_comet_score}")
        return avg_comet_score

    except Exception as e:
        logging.error(f"Error computing COMET score: {str(e)}")
        return None
