from datasketch import MinHash


def get_5grams(s):
    words = s.split()
    return [" ".join(words[i : i + 5]) for i in range(len(words) - 4)]


def compute_minhash_from_text(text):
    m = MinHash(num_perm=256)
    for gram in get_5grams(text):
        m.update(gram.encode("utf8"))
    return m


def compute_jaccard_similarity(text1, text2):
    minhash1 = compute_minhash_from_text(text1)
    minhash2 = compute_minhash_from_text(text2)
    return minhash1.jaccard(minhash2)
