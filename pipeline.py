import logging
import os

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer

from configuration import ExperimentConfig
from generation import CodeGenerator
from perplexity import Perplexity

logger = logging.getLogger(__name__)


class ExperimentPipeline:
    def __init__(self, path_to_yaml: str):
        self.config = ExperimentConfig.from_yaml(path_to_yaml)

    def run(self) -> None:
        logger.info(
            f"Running experiment {self.config.experiment_name} with model {self.config.model_load.model_name}"
        )
        logger.info(f"Code generation config: {self.config.code_generation}")
        logger.info(f"Perplexity config: {self.config.perplexity}")
        logger.info(f"Save path: {self.config.save_path}")
        experiment_results = []

        logger.info("Loading the model and tokenizer")
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_load.model_name,
            device_map="auto",
            quantization_config=self.config.model_load.quantization_config,
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
            languages = df["language"].unique()
            for language in languages:
                logger.info("Generating code for language: {language}")
                prompts = df[df["language"] == language]["prompt"].tolist()
                results = generator.generate_text_with_chat_template(
                    prompts, language=language
                )
                experiment_results += [
                    {
                        "language": language,
                        "prompt": result.prompt,
                        "generated_code": result.generated_code,
                        "prediction": result.complete_code,
                    }
                    for result in results
                ]

        if self.config.perplexity:
            logger.info("Running perplexity calculation")
            # use the generation results to calculate perplexity if code generation step is run before

            if self.config.code_generation is None:
                # if code generation step is not run, use the prompts from the dataset
                # predictions column is mandatory for perplexity calculation prompts can be skipped
                assert len(experiment_results) == 0
                experiment_results += [
                    {"prediction": prediction, "prompt": prompt}
                    for prediction, prompt in zip(
                        self.config.dataset["predictions"].tolist(),
                        (
                            self.config.dataset["prompts"].tolist()
                            if "prompts" in self.config.dataset.columns
                            else [None] * len(self.config.dataset["predictions"])
                        ),
                    )
                ]

            perplexity = Perplexity(
                model=model,
                tokenizer=tokenizer,
                device=self.config.model_load.device,
                batch_size=self.config.perplexity.batch_size,
            )

            perplexity_results = perplexity.compute(
                prompts=[result["prompt"] for result in experiment_results],
                predictions=[result["prediction"] for result in experiment_results],
                thresholds=self.config.perplexity.thresholds,
            )

            token_probabilities = perplexity_results[
                f"{self.config.perplexity.thresholds[0]}"
            ]["probs"]
            mean_perplexities = []
            filtered_token_percentages = []
            for threshold, results in perplexity_results.items():
                mean_perplexities.append(results["mean_perplexity"])
                filtered_token_percentages.append(results["filtered_token_percentage"])

                for item, perp, ls in zip(
                    experiment_results,
                    results["perplexities"],
                    results["longest_filtered_sequences"],
                ):
                    item[f"{threshold}_perplexity"] = perp
                    item[f"{threshold}_longest_filtered_sequence"] = ls

            for item, gp in zip(
                experiment_results,
                perplexity_results[str(self.config.perplexity.thresholds[0])][
                    "sample_probs"
                ],
            ):
                item["token_probabilities"] = gp

            # create the save path if it does not exist
            os.makedirs(self.config.save_path, exist_ok=True)

            # save the results
            df = pd.DataFrame.from_dict(experiment_results)
            df.to_csv(self.config.save_path + "/results.csv", index=False)
            logger.info(f"Results saved at {self.config.save_path}/results.csv")

            # draw plots of probability distribution, mean perplexity and filtered token percentage
            sns.kdeplot(token_probabilities)
            plt.xlabel("Token Probability")
            plt.ylabel("Density")
            plt.title("Token Probability Distribution")
            plt.savefig(self.config.save_path + "/token_probability_distribution.png")
            plt.close()

            if 1.01 in self.config.perplexity.thresholds:
                # remove the 1.01 threshold if it exists
                self.config.perplexity.thresholds[
                    self.config.perplexity.thresholds.index(1.01)
                ] = 1
            # plot the mean perplexity by threshold
            plt.plot(self.config.perplexity.thresholds, mean_perplexities)
            plt.xlabel("Threshold")
            plt.ylabel("Mean Perplexity")
            plt.title("Mean Perplexity by Threshold")
            plt.gca().invert_xaxis()
            plt.grid(True)
            plt.savefig(self.config.save_path + "/mean_perplexity_by_threshold.png")
            plt.close()

            # plot the filtered token percentage by threshold
            plt.plot(self.config.perplexity.thresholds, filtered_token_percentages)
            plt.xlabel("Threshold")
            plt.ylabel("Filtered Token Percentage")
            plt.title("Filtered Token Percentage by Threshold")
            plt.gca().invert_xaxis()
            plt.grid(True)
            plt.savefig(
                self.config.save_path + "/filtered_token_percentage_by_threshold.png"
            )
