from similarity.bleu import compute_bleu_score


class TestBleuScore:
    def test_bleu_returns_1_when_strings_are_similar(self):
        str1 = "Hello world this is a test"
        str2 = "Hello world this is a test"
        assert compute_bleu_score(str1, str2) == 1

    def test_bleu_distance_0_when_strings_are_different(self):
        str1 = "Hello World!"
        str2 = "This is a random string"
        assert compute_bleu_score(str1, str2) == 0
