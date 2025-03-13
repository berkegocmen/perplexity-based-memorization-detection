from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


def compute_bleu_score(reference_text, generated_text):
    reference_text_list = [reference_text.split()]
    generated_text_list = generated_text.split()
    return sentence_bleu(reference_text_list, generated_text_list, smoothing_function=SmoothingFunction().method4)
