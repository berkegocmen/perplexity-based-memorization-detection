import itertools
import logging
import os

import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from configuration import ExperimentConfig
from generation import CodeGenerator
from perplexity import Perplexity
from perplexity.tfidf import compute_tfidf_for_vocab

logger = logging.getLogger(__name__)


class ExperimentPipeline:
    def __init__(self, path_to_yaml: str):
        self.config = ExperimentConfig.from_yaml(path_to_yaml)

    def run(self) -> None:
        logger.info(f"Running experiment {self.config.experiment_name} with model {self.config.model_load.model_name}")
        logger.info(f"Code generation config: {self.config.code_generation}")
        logger.info(f"Perplexity config: {self.config.perplexity}")
        logger.info(f"Save path: {self.config.save_path}")

        logger.info("Loading the model and tokenizer")
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_load.model_name,
            device_map="auto",
            quantization_config=self.config.model_load.quantization_config,
            torch_dtype=torch.bfloat16,
        )
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_load.model_name)

        if self.config.code_generation:
            logger.info("Running code generation")
            # code generation logic here
            generator = CodeGenerator(
                model=model,
                tokenizer=tokenizer,
                generation_config=self.config.code_generation.generation_config,
                generation_template=self.config.code_generation.chat_template,
                batch_size=self.config.code_generation.batch_size,
            )

            # expected columns in the dataset ["prompt", "language"]
            df = self.config.dataset
            df["prediction"] = None
            df["generated_code"] = None
            languages = df["language"].unique()
            for language in languages:
                logger.info(f"Generating code for language: {language}")
                prompts = df[df["language"] == language]["prompt"].tolist()
                results = generator.generate_text_with_chat_template(prompts, language=language)

                for idx, result in zip(df[df["language"] == language].index, results):
                    assert df.at[idx, "prompt"] == result.prompt
                    df.at[idx, "prediction"] = result.complete_code
                    df.at[idx, "generated_code"] = result.generated_code
            self.config.dataset = df
            df.to_csv(self.config.save_path + "generated_code.csv", index=False)

        if self.config.perplexity:
            logger.info("Running perplexity calculation")
            # use the generation results to calculate perplexity if code generation step is run before
            tfidf = None
            if self.config.perplexity.tfidf_enabled:
                # calculate tfidf for the dataset
                logger.info("Calculating TF-IDF for the dataset")
                tfidf = compute_tfidf_for_vocab(self.config.dataset["prediction"].tolist(), tokenizer)

            perplexity = Perplexity(
                model=model,
                tokenizer=tokenizer,
                device=self.config.model_load.device,
                batch_size=self.config.perplexity.batch_size,
                tfidf=tfidf,
            )

            perplexity_results = perplexity.compute(
                prompts=self.config.dataset["prompt"].tolist(),
                predictions=self.config.dataset["prediction"].tolist(),
                thresholds=self.config.perplexity.thresholds,
            )

            new_columns = list(
                itertools.chain.from_iterable(
                    [
                        [f"{threshold}_perplexity", f"{threshold}_longest_filtered_sequence"]
                        for threshold in self.config.perplexity.thresholds
                    ]
                )
            ) + ["token_probabilities"]

            # add the columns to the dataset
            for column in new_columns:
                self.config.dataset[column] = None

            filtered_token_percentages = []

            for threshold, results in perplexity_results.items():
                filtered_token_percentages.append(results["filtered_token_percentage"])

                for idx, perp, ls in zip(
                    self.config.dataset.index, results["perplexities"], results["longest_filtered_sequences"]
                ):
                    self.config.dataset.at[idx, f"{threshold}_perplexity"] = perp
                    self.config.dataset.at[idx, f"{threshold}_longest_filtered_sequence"] = ls

            for idx, gp in zip(
                self.config.dataset.index, perplexity_results[str(self.config.perplexity.thresholds[0])]["sample_probs"]
            ):
                self.config.dataset.at[idx, "token_probabilities"] = gp

            source_filtered_token_percentages = []
            if "source" in self.config.dataset.columns:
                source_perplexity_results = perplexity.compute(
                    prompts=self.config.dataset["prompt"].tolist(),
                    predictions=self.config.dataset["source"].tolist(),
                    thresholds=self.config.perplexity.thresholds,
                )

                new_columns = list(
                    itertools.chain.from_iterable(
                        [
                            [f"{threshold}_source_perplexity", f"{threshold}_source_longest_filtered_sequence"]
                            for threshold in self.config.perplexity.thresholds
                        ]
                    )
                ) + ["source_token_probabilities"]

                # add the columns to the dataset
                for column in new_columns:
                    self.config.dataset[column] = None

                for threshold, results in source_perplexity_results.items():
                    source_filtered_token_percentages.append(results["filtered_token_percentage"])

                    for idx, perp, ls in zip(
                        self.config.dataset.index, results["perplexities"], results["longest_filtered_sequences"]
                    ):
                        self.config.dataset.at[idx, f"{threshold}_source_perplexity"] = perp
                        self.config.dataset.at[idx, f"{threshold}_source_longest_filtered_sequence"] = ls

                for idx, gp in zip(
                    self.config.dataset.index,
                    source_perplexity_results[str(self.config.perplexity.thresholds[0])]["sample_probs"],
                ):
                    self.config.dataset.at[idx, "source_token_probabilities"] = gp

            # create the save path if it does not exist
            os.makedirs(self.config.save_path, exist_ok=True)

            # save the results
            self.config.dataset.to_csv(self.config.save_path + "/results.csv", index=False)
            logger.info(f"Results saved at {self.config.save_path}/results.csv")
            # save the source_filtered_token_percentages and filtered_token_percentages into a json file
            if len(source_filtered_token_percentages) != len(filtered_token_percentages):
                source_filtered_token_percentages = [None] * len(filtered_token_percentages)
            pd.DataFrame(
                {
                    "source_filtered_token_percentages": source_filtered_token_percentages,
                    "filtered_token_percentages": filtered_token_percentages,
                }
            ).to_json(self.config.save_path + "/token_percentages.json")
