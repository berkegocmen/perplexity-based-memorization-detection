from transformers import GenerationConfig

from configuration import (
    ExperimentConfig,
    ModelLoadConfig,
    CodeGenerationConfig,
    PerplexityConfig,
)


class TestConfiguration:
    def test_configuration(self):
        configuration = ExperimentConfig.from_yaml("configs/test_config.yaml")

        assert configuration.experiment_name == "test"

        assert isinstance(configuration.model_load, ModelLoadConfig)
        assert configuration.model_load.quantization_config is None
        assert configuration.model_load.model_name == "codeparrot/codeparrot"

        assert isinstance(configuration.code_generation, CodeGenerationConfig)
        assert configuration.code_generation.batch_size == 1
        assert configuration.code_generation.chat_template is None
        assert isinstance(
            configuration.code_generation.generation_config, GenerationConfig
        )
        assert configuration.code_generation.generation_config.max_new_tokens == 512

        assert isinstance(configuration.perplexity, PerplexityConfig)
        assert configuration.perplexity.batch_size == 1
        assert configuration.perplexity.thresholds == [
            1.01,
            0.999,
            0.99,
            0.98,
            0.97,
            0.96,
            0.95,
            0.9,
            0.8,
            0.7,
            0.6,
            0.5,
        ]

        assert configuration.save_path == "tests/data/test_experiment"
