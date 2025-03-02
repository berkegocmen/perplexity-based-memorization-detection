from unittest.mock import MagicMock, patch

from transformers import GenerationConfig, AutoTokenizer, AutoModelForCausalLM

from generation import CodeGenerator
from perplexity.base import GeneratedCode


class TestCodeGenerator:
    def test_generation_with_template(self):
        mock_tokenizer = AutoTokenizer.from_pretrained(
            "hf-internal-testing/tiny-random-BloomForCausalLM"
        )
        mock_model = AutoModelForCausalLM.from_pretrained(
            "hf-internal-testing/tiny-random-BloomForCausalLM"
        )
        mock_generation_config = GenerationConfig(
            max_new_tokens=5, pad_token_id=mock_tokenizer.eos_token_id, device="cpu"
        )

        generator = CodeGenerator(
            model=mock_model,
            tokenizer=mock_tokenizer,
            generation_config=mock_generation_config,
        )

        with patch.object(
            generator,
            "generation_pipeline",
            return_value=[
                [
                    {
                        "generated_text": """### Response\nHere's how you can complete the function:\n```python\ndef hello_world():\n    print('Hello, World!')"""
                    }
                ]
            ],
        ):
            result = generator.generate_text_with_chat_template(
                ["def hello_world():"], language="python"
            )
            assert isinstance(result[0], GeneratedCode)
            assert result[0].prompt == "def hello_world():"
            assert result[0].generated_code == "\n    print('Hello, World!')"
            assert (
                result[0].complete_code
                == "def hello_world():\n    print('Hello, World!')"
            )
