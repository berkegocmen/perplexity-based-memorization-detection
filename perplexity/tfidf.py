from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


def compute_tfidf_for_vocab(dataset, tokenizer, aggregation_fn=np.mean):
    """
    Compute TF-IDF values for the tokenizer's vocabulary based on the dataset.

    :param dataset: List of text samples (training set).
    :param tokenizer: Tokenizer used by the LLM.
    :param aggregation_fn: Function to aggregate TF-IDF values for sub-tokens (e.g., np.mean, np.sum, np.max).
    :return: Dictionary mapping token IDs to their aggregated TF-IDF values.
    """
    # Tokenize the dataset into strings of tokens
    tokenized_texts = [" ".join(map(str, tokenizer.encode(text, add_special_tokens=False))) for text in dataset]

    # Use TfidfVectorizer to compute TF-IDF
    vectorizer = TfidfVectorizer()
    vectorizer.fit(tokenized_texts)

    # Map token IDs to their aggregated TF-IDF values
    tfidf_values = {}
    for token, idx in tokenizer.get_vocab().items():
        token_ids = tokenizer.encode(token, add_special_tokens=False)
        tfidf_scores = [
            vectorizer.idf_[vectorizer.vocabulary_[str(sub_token)]]
            for sub_token in token_ids
            if str(sub_token) in vectorizer.vocabulary_
        ]
        if tfidf_scores:
            tfidf_values[idx] = aggregation_fn(tfidf_scores)  # Aggregate using the specified function
        else:
            tfidf_values[idx] = 0.001  # Assign a default value if no sub-tokens are found

    return tfidf_values
