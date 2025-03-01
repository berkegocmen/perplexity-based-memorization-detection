import gc
import re

import torch
import weave
from transformers import (
    pipeline,
    GenerationConfig,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    AutoTokenizer,
)

from perplexity.base import GeneratedCode
from evaluate import logging

from utils import ListDataset

DEFAULT_CHAT_TEMPLATE = """{{{{ bos_token }}}} You are an exceptionally intelligent coding assistant that consistently delivers accurate and reliable responses to user instructions.

### Instruction
Complete the function in the code snippet:
```{language}
{{{{ messages }}}}
```

### Response
Here's how you can complete the function:
```{language}
{{{{ messages }}}}
"""


class CodeGenerator(weave.Model):
    model_name: str
    quantization_config: BitsAndBytesConfig | None = None
    generation_config: GenerationConfig

    def __init__(
        self,
        model_name: str,
        quantization_config: BitsAndBytesConfig,
        generation_config: GenerationConfig,
        generation_template: str | None = None,
        batch_size=1,
    ) -> None:
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=quantization_config
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.generation_config = generation_config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.genereration_template = generation_template or DEFAULT_CHAT_TEMPLATE
        self.batch_size = batch_size

        # initialize the generation pipeline
        self.generation_pipeline = pipeline(
            model=self.model,
            task="text-generation",
            device_map="auto",
            tokenizer=self.tokenizer,
            framework="pt",
        )

    def generate_code(self, prompt) -> str:
        # Tokenize and get both input_ids and attention_mask
        encodings = self.tokenizer(
            prompt, return_tensors="pt", return_attention_mask=True
        ).to(self.device)
        input_ids = encodings["input_ids"]
        attention_mask = encodings["attention_mask"]
        outputs = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            generation_config=self.generation_config,
            tokenizer=self.tokenizer,
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    @weave.op(name="generate_code")
    def generate_text_with_chat_template(
        self, prompts: list[str], language: str
    ) -> list[GeneratedCode]:
        results = []
        # regex pattern to extract the code from the generated text
        pattern = rf"```{language}\n(.*?)\n```"

        # apply the chat template to the tokenizer and the prompts
        self.tokenizer.chat_template = self.genereration_template.format(
            language=language
        )

        teminators = [
            self.generation_pipeline.tokenizer.eos_token_id,
            self.generation_pipeline.tokenizer.convert_tokens_to_ids("###"),
        ]

        # apply the chat template to the prompts
        instructions = [
            self.generation_pipeline.tokenizer.apply_chat_template(
                prompt, tokenize=False, add_generation_prompt="True"
            )
            for prompt in prompts
        ]

        for idx, (result, prompt) in enumerate(
            zip(
                logging.tqdm(
                    self.generation_pipeline(
                        ListDataset(instructions),
                        generation_config=self.generation_config,
                        eos_token_id=teminators,
                        pad_token_id=self.generation_pipeline.tokenizer.eos_token_id,
                        batch_size=self.batch_size,
                    ),
                    desc="Generating text",
                ),
                prompts,
            )
        ):
            # extract the code from the generated text
            response = result[0]["generated_text"].split("### Response")[-1]
            # separate prompt and the generated code
            matches = re.findall(
                pattern,
                response,
                re.DOTALL,
            )
            if len(matches) > 0:
                code = matches[0]
            else:
                code = response.split(
                    "Here's how you can complete the function:\n```python"
                )[-1].lstrip()

            results.append(
                GeneratedCode(
                    prompt=prompt, generated_code=code[len(prompt.lstrip()) :]
                )
            )

            # collect garbage
            gc.collect()
            torch.cuda.empty_cache()

        return results
