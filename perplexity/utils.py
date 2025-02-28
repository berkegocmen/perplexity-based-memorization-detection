import json

import torch


# load data
def load_json_file(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data


def to_tokens_and_probs(model, tokenizer, input_texts: list[str], as_log_probs: bool = False) -> list[
    list[tuple[str, float]]]:
    input_ids = tokenizer(input_texts, padding=True, return_tensors="pt").input_ids
    outputs = model(input_ids)
    if as_log_probs:
        probs = torch.log_softmax(outputs.logits, dim=-1).detach()
    else:
        probs = torch.softmax(outputs.logits, dim=-1).detach()

    # collect the probability of the generated token -- probability at index 0 corresponds to the token at index 1
    probs = probs[:, :-1, :]
    input_ids = input_ids[:, 1:]
    gen_probs = torch.gather(probs, 2, input_ids[:, :, None]).squeeze(-1)

    batch = []
    for input_sentence, input_probs in zip(input_ids, gen_probs):
        text_sequence = []
        for token, p in zip(input_sentence, input_probs):
            if token not in tokenizer.all_special_ids:
                text_sequence.append((tokenizer.decode(token), p.item()))
        batch.append(text_sequence)
    return batch
