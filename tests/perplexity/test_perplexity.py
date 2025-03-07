import numpy as np
import pytest
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from perplexity import Perplexity


class TestPerplexity:

    def test_pad_tensors_return_correct_shape(self):
        # mock logits, labels, and attention mask
        logits = [torch.randn(2, 10), torch.randn(5, 10), torch.randn(3, 10)]
        labels = [torch.rand(2), torch.rand(5), torch.rand(3)]
        attn_mask = [
            torch.ones(2, dtype=torch.int),
            torch.ones(5, dtype=torch.int),
            torch.ones(3, dtype=torch.int),
        ]

        # initialize mock model and tokenizer
        mock_tokenizer = AutoTokenizer.from_pretrained(
            "hf-internal-testing/tiny-random-BloomForCausalLM"
        )
        mock_model = AutoModelForCausalLM.from_pretrained(
            "hf-internal-testing/tiny-random-BloomForCausalLM"
        )
        perp = Perplexity(model=mock_model, tokenizer=mock_tokenizer)

        padded_logits, padded_labels, padded_attn_mask = perp._pad_tensors(
            logits, labels, attn_mask
        )

        assert padded_logits.shape == (3, 5, 10)
        assert padded_labels.shape == (3, 5)
        assert padded_attn_mask.shape == (3, 5)

    def test_threshold_filtering(self):
        # mock logits with number so that it will map to high probability
        logits = torch.tensor(
            [
                [
                    [0.1, 0.2, 0.3, 0.4, 0.5],
                    [0.6, 0.7, 0.8, 0.9, 0.1],
                    [0.2, 0.3, 0.4, 0.5, 0.6],
                ],
                [
                    [0.1, 0.2, 0.3, 0.4, 0.5],
                    [0.6, 0.7, 0.8, 0.9, 0.1],
                    [0.2, 0.3, 0.4, 0.5, 0.6],
                ],
            ]
        )
        labels = torch.tensor([[1, 3, 4], [1, 3, 4]])
        attn_mask = torch.tensor([[1, 1, 1], [1, 1, 1]])

        # initialize mock model and tokenizer
        mock_tokenizer = AutoTokenizer.from_pretrained(
            "hf-internal-testing/tiny-random-BloomForCausalLM"
        )
        mock_model = AutoModelForCausalLM.from_pretrained(
            "hf-internal-testing/tiny-random-BloomForCausalLM"
        )
        perp = Perplexity(model=mock_model, tokenizer=mock_tokenizer)

        threshold = 0.2
        (
            (filtered_logits, filtered_labels, filtered_attn_mask),
            probs_collection,
            tt,
            ft,
            longest_sequences,
            gp,
        ) = perp._filter_on_threshold(logits, labels, attn_mask, threshold)

        assert filtered_logits.shape == (2, 1, 5)
        assert filtered_labels.shape == (2, 1)
        assert filtered_attn_mask.shape == (2, 1)
        assert tt == [2, 2]
        assert ft == [1, 1]
        assert longest_sequences == pytest.approx([0.5, 0.5])
        assert isinstance(gp, list)
        assert len(gp) == 2

    def test_compute_runs_without_error(self):
        # initialize mock model and tokenizer
        mock_tokenizer = AutoTokenizer.from_pretrained(
            "hf-internal-testing/tiny-random-BloomForCausalLM"
        )
        mock_model = AutoModelForCausalLM.from_pretrained(
            "hf-internal-testing/tiny-random-BloomForCausalLM"
        )
        perp = Perplexity(model=mock_model, tokenizer=mock_tokenizer)

        prompts = ["def hello_world():", "def hello_world():", "def hello_world():"]
        predictions = [
            "def hello_world():\n    print('Hello, World!')",
            "def hello_world():\n    print('Hello, World!')",
            "def hello_world():\n    print('Hello, World!')",
        ]

        results = perp.compute(prompts, predictions)
        assert type(results["1.01"]["mean_perplexity"]) == np.float64
        assert len(results["1.01"]["perplexities"]) == 3
        assert type(results["1.01"]["filtered_token_percentage"]) == float
        assert len(results["1.01"]["longest_filtered_sequences"]) == 3
        assert len(results["1.01"]["sample_probs"]) == 3
