from transformers import AutoTokenizer

from similarity.edit_distance import compute_edit_distance


class TestEditDistance:
    def test_edit_distance_returns_0_when_strings_are_similar(self):
        str1 = "Hello World!"
        str2 = "Hello World!"
        tokenizer = AutoTokenizer.from_pretrained("bigcode/tiny_starcoder_py")
        assert compute_edit_distance(str1, str2, tokenizer) == 0

    def test_edit_distance_returns_1_when_strings_are_different(self):
        str1 = "Hello World!"
        str2 = "This is a random string"
        tokenizer = AutoTokenizer.from_pretrained("bigcode/tiny_starcoder_py")
        assert compute_edit_distance(str1, str2, tokenizer) == 1
