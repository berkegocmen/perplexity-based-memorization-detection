from transformers import PreTrainedTokenizer


def levenshtein_distance_star(str1, str2):
    if len(str1) > len(str2):
        str1, str2 = str2, str1

    distances = range(len(str1) + 1)
    for index2, char2 in enumerate(str2):
        new_distances = [index2 + 1]
        for index1, char1 in enumerate(str1):
            if char1 == char2:
                new_distances.append(distances[index1])
            else:
                new_distances.append(1 + min((distances[index1], distances[index1 + 1], new_distances[-1])))
        distances = new_distances

    return distances[-1]


def compute_edit_distance(candidate: str, reference: str, tokenizer: PreTrainedTokenizer) -> float:
    reference = reference.strip()
    r = tokenizer.encode(reference, add_special_tokens=False)
    candidate = candidate.strip()
    c = tokenizer.encode(candidate, add_special_tokens=False)
    return levenshtein_distance_star(c, r) / max(len(c), len(r)) if max(len(c), len(r)) > 0 else 1
