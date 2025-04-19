import pandas as pd
from transformers import AutoTokenizer

from perplexity.tfidf import compute_tfidf_for_vocab


class TestTFIDF:

    def test_tfidf_with_test_dataset(self):
        df = pd.read_csv("tests/data/test.csv")
        dataset = df["source"].tolist()
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-BloomForCausalLM")
        tfidf_values = compute_tfidf_for_vocab(dataset, tokenizer)
        assert isinstance(tfidf_values, dict)
