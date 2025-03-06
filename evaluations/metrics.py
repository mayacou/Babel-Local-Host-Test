# evaluations/metrics.py

import sacrebleu


def compute_bleu(generated_translations, reference_translations):
    bleu = sacrebleu.corpus_bleu(generated_translations, [reference_translations])
    return bleu.score

from comet import download_model, load_from_checkpoint

import evaluate

import evaluate

def compute_comet(generated_translations, reference_translations, source_sentences):
    comet = evaluate.load("comet")

    results = comet.compute(
        predictions=generated_translations,
        references=reference_translations,
        sources=source_sentences
    )

    return results["mean_score"]
