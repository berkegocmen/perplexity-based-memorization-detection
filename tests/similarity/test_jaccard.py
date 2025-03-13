from similarity.jaccard import compute_jaccard_similarity


class TestJaccardSimilarity:

    def test_jaccard_similarity_returns_1_when_strings_are_similar(self):
        str1 ="Hello World!"
        str2 = "Hello World!"
        assert compute_jaccard_similarity(str1, str2) ==1

    def test_jaccard_similarity_returns_0_when_strings_are_different(self):
        str1 ="Hello World!"
        str2 = "This is a random string"
        assert compute_jaccard_similarity(str1, str2) ==0