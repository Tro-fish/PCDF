import argparse
import sacrebleu

def bleu(targets, predictions, tokenizer="intl"):
  """Computes BLEU score.

  Args:
    targets: list of strings or list of list of strings if multiple references
      are present.
    predictions: list of strings
    tokenizer: tokenizer option for corpus_bleu

  Returns:
    bleu_score across all targets and predictions
  """
  if isinstance(targets[0], list):
    targets = [[x for x in target] for target in targets]
  else:
    # Need to wrap targets in another list for corpus_bleu.
    targets = [targets]

  bleu_score = sacrebleu.corpus_bleu(predictions, targets,
                                     smooth_method="exp",
                                     smooth_value=0.0,
                                     force=False,
                                     lowercase=False,
                                     tokenize=tokenizer,
                                     use_effective_order=False)
  return {"bleu": bleu_score.score}



pred_path = "output/pew/pred_answers.txt"
ref_path = "output/pew/true_answers.txt" 

with open(pred_path, 'r', encoding='utf-8') as pred:
    predictions = [p.strip() for p in pred.readlines()]

with open(ref_path, 'r', encoding='utf-8') as ref:
    references = [r.strip() for r in ref.readlines()]

BLEU_SCORE = bleu(references, predictions)['bleu']
print(BLEU_SCORE)