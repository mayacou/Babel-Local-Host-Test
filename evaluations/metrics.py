import sacrebleu
from comet import download_model, load_from_checkpoint
import logging
import re

# Set up logging for errors and info
logging.basicConfig(level=logging.INFO)

import torch
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

# Helper to remove hypothesis output
def clean_output(text):
    # Remove special tokens like [INST], <s>, and </s>
    text = re.sub(r"\[INST\].*?\[\/INST\]", "", text, flags=re.DOTALL)
    text = re.sub(r"<s>|</s>", "", text)

    # Remove any lingering translation instruction like Translate this to [language]
    text = re.sub(r"\[INST\]\s*Translate this to \w+:", "", text, flags=re.IGNORECASE)

    # Ensure that the instruction isn't reintroduced
    text = re.sub(r"\[INST\]\s*Translate this to \w+:\s*", "", text, flags=re.IGNORECASE)

    # Remove excessive whitespace and line breaks
    text = re.sub(r"\s+", " ", text).strip()

    return text



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
    """Compute BLEU score using SacreBLEU with normalization and debugging."""
    try:
        # Clean the model output before processing
        print("BEFORE CLEAN HYPOTHESIS -> ", hypotheses)
        hypotheses = [clean_output(hyp) for hyp in hypotheses]
        
        # Normalize both references and hypotheses
        references = [ref.strip().lower() for ref in references]

        # Debugging: Print sample references and hypotheses
        print(f"Sample References: {references[:3]}")
        print(f"Sample Hypotheses: {hypotheses[:3]}")

        # Check for empty hypotheses
        if not any(hypotheses):
            logging.warning("All hypotheses are empty! BLEU score will be low.")
            return 0.0

        # Compute BLEU with default '13a' tokenization
        bleu = sacrebleu.corpus_bleu(hypotheses, [references], tokenize='13a')
        logging.info(f"BLEU score: {bleu.score}")

        # Compute chrF (useful if BLEU is low)
        chrf_score = sacrebleu.corpus_chrf(hypotheses, [references]).score
        logging.info(f"chrF Score: {chrf_score}")

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
        # Clean the model output before processing
        hypotheses = [clean_output(hyp) for hyp in hypotheses]

        # Ensure the sources, hypotheses, and references are aligned correctly
        if len(references) != len(hypotheses) or len(hypotheses) != len(sources):
            logging.error("Mismatched input lengths: references, hypotheses, and sources must be of the same length.")
            return None

        # Format the data for COMET
        data = [{"src": src, "mt": hyp, "ref": ref} for src, hyp, ref in zip(sources, hypotheses, references)]
        comet_scores = comet_model.predict(data, batch_size=batch_size)

        # Log individual scores for debugging
        for idx, score in enumerate(comet_scores.scores):
            logging.info(f"COMET score for index {idx}: {score}")

        # Return the average score
        avg_comet_score = sum(comet_scores.scores) / len(comet_scores.scores) if comet_scores.scores else None
        logging.info(f"COMET score: {avg_comet_score}")
        return avg_comet_score

    except Exception as e:
        logging.error(f"Error computing COMET score: {str(e)}")
        return None