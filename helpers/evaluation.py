import nltk
import sacrebleu
from comet.models import download_model, load_from_checkpoint

# Download and load COMET model (for more accurate evaluation)
COMET_MODEL_PATH = download_model("Unbabel/wmt22-comet-da")
comet_model = load_from_checkpoint(COMET_MODEL_PATH)

def compute_bleu(references, hypotheses):
    """Compute BLEU score using SacreBLEU"""
    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    return bleu.score

def compute_comet(references, hypotheses, sources):
    """Compute COMET score for more accurate evaluation"""
    data = [{"src": src, "mt": hyp, "ref": ref} for src, hyp, ref in zip(sources, hypotheses, references)]
    comet_scores = comet_model.predict(data, batch_size=8)
    return sum(comet_scores.scores) / len(comet_scores.scores)  # Average score
