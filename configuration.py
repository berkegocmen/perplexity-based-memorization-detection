import pandas as pd
import torch
import yaml
from pandas import DataFrame
from pydantic import BaseModel, ConfigDict
from transformers import GenerationConfig, BitsAndBytesConfig


class ConfigBaseModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)


class ModelLoadConfig(ConfigBaseModel):
    model_name: str
    quantization_config: BitsAndBytesConfig | None = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class CodeGenerationConfig(ConfigBaseModel):
    generation_config: GenerationConfig
    chat_template: str | None = None
    batch_size: int = 1


class PerplexityConfig(ConfigBaseModel):
    thresholds: list[float | int]
    batch_size: int = 1


class ExperimentConfig(ConfigBaseModel):
    experiment_name: str
    model_load: ModelLoadConfig
    dataset: DataFrame
    code_generation: CodeGenerationConfig | None = None
    perplexity: PerplexityConfig | None = None
    save_path: str

    # load the config from yaml file
    @classmethod
    def from_yaml(cls, path: str):
        with open(path, "r") as f:
            config = yaml.safe_load(f)

        return ExperimentConfig(
            experiment_name=config["experiment_name"],
            model_load=ModelLoadConfig(
                model_name=config["model_load"]["model_name"],
                quantization_config=(
                    BitsAndBytesConfig(**config["model_load"]["quantization"]["params"])
                    if config["model_load"]["quantization"]["enabled"]
                    else None
                ),
                device=config["model_load"].get("device", None),
            ),
            code_generation=(
                CodeGenerationConfig(
                    batch_size=config["code_generation"]["params"]["batch_size"],
                    generation_config=(GenerationConfig(**config["code_generation"]["params"]["generation_config"])),
                    chat_template=(
                        # if chat_template exist in the keys else None
                        config["code_generation"].get("chat_template", None)
                    ),
                )
                if config["code_generation"]["enabled"]
                else None
            ),
            perplexity=(
                PerplexityConfig(
                    batch_size=config["perplexity"]["params"]["batch_size"],
                    thresholds=config["perplexity"]["params"]["thresholds"],
                )
                if config["perplexity"]["enabled"]
                else None
            ),
            dataset=pd.read_csv(config["dataset"]),
            save_path=config["save_path"],
        )

