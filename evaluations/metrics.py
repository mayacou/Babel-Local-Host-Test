# evaluations/metrics.py

import sacrebleu


def compute_bleu(generated_translations, reference_translations):
    bleu = sacrebleu.corpus_bleu(generated_translations, [reference_translations])
    return bleu.score

from comet import download_model, load_from_checkpoint

import evaluate

def compute_comet(generated_translations, reference_translations):
    comet = evaluate.load("comet")
    results = comet.compute(predictions=generated_translations, references=reference_translations)
    return results["mean_score"]
