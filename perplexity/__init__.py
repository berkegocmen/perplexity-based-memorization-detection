import numpy as np
from torch import Tensor
from transformers import GenerationConfig, pipeline
import torch
from torch.nn import CrossEntropyLoss
from tqdm.auto import tqdm
import gc
from torch.utils.data import Dataset
import re
from evaluate import logging

DEFAULT_CHAT_TEMPLATE = """{{ bos_token }} You are an exceptionally intelligent coding assistant that consistently delivers accurate and reliable responses to user instructions.

### Instruction
Complete the function in the code snippet:
python
```
{{ messages }}
```

### Response
Here's how you can complete the function:
```python
{{ messages }}
"""


class ListDataset(Dataset):
    def __init__(self, original_list):
        self.original_list = original_list

    def __len__(self):
        return len(self.original_list)

    def __getitem__(self, i):
        return self.original_list[i]

    def __iter__(self):
        """Returns an iterator over the dataset."""
        for item in self.original_list:
            yield item

    def __repr__(self):
        return repr(self.original_list)


class Perplexity:
    def __init__(
            self,
            model,
            tokenizer,
            device,
            use_chat_template: bool,
            generation_config: None | GenerationConfig,
            batch_size: int = 1,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token

        if device is not None:
            assert device in [
                "gpu",
                "cpu",
                "cuda",
            ], "device should be either gpu or cpu."
            if device == "gpu":
                device = "cuda"
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.generation_config = (
            generation_config
            if generation_config
            else GenerationConfig(
                max_new_tokens=512,  # Maximum number of tokens to generate
                pad_token_id=tokenizer.eos_token_id,
                tokenizer=tokenizer,
                add_special_tokens=False,
                use_cache=False,
            )
        )

        self.batch_size = batch_size
        self.loss_fct = CrossEntropyLoss(reduction="none")
        self.generation_pipeline = None
        self.apply_chat_template()
        self.use_chat_template = use_chat_template

    def generate_text(self, prompt):
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

    def apply_chat_template(self, template: str | None = None):
        if template:
            self.tokenizer.chat_template = template
            return
        self.tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE

    def generate_text_with_chat_template(self, prompts):
        pattern = r"```python\n(.*?)\n```"
        if self.generation_pipeline is None:
            self.generation_pipeline = pipeline(
                model=self.model,
                task="text-generation",
                device_map="auto",
                tokenizer=self.tokenizer,
                framework="pt",
            )

        teminators = [
            self.generation_pipeline.tokenizer.eos_token_id,
            self.generation_pipeline.tokenizer.convert_tokens_to_ids("###"),
        ]

        instructions = [
            self.generation_pipeline.tokenizer.apply_chat_template(
                prompt, tokenize=False, add_generation_prompt="True"
            )
            for prompt in prompts
        ]

        results = []
        responses = []

        for idx, result in enumerate(
                tqdm(
                    self.generation_pipeline(
                        ListDataset(instructions),
                        generation_config=self.generation_config,
                        eos_token_id=teminators,
                        pad_token_id=self.generation_pipeline.tokenizer.eos_token_id,
                        batch_size=1,
                    ),
                    desc="Generating text",
                )
        ):
            results.append(result)
            print("Generated Text --->")
            matches = re.findall(
                pattern,
                result[0]["generated_text"].split("### Response")[-1],
                re.DOTALL,
            )
            if len(matches) > 0:
                responses.append(matches[0])
                print(matches[0])
            else:
                responses.append(
                    result[0]["generated_text"].split(
                        "### Response\nHere's how you can complete the function:\n```python"
                    )[-1]
                )
                print(
                    result[0]["generated_text"].split(
                        "### Response\nHere's how you can complete the function:\n```python"
                    )[-1]
                )
            gc.collect()  # Collect garbage at the end of each batch
            torch.cuda.empty_cache()

        return responses

    def compute_perplexity(self, prompts: list[str]):
        ppls = []
        # get predictions one by one
        if self.use_chat_template:
            predictions = self.generate_text_with_chat_template(prompts)
        else:
            for prompt in tqdm(prompts, desc="Generating predictions for the prompts"):
                predictions = []

                predictions.append(
                    self.generate_text(prompt)
                )  # containes prompt + generated_text

        for start_index in tqdm(
                range(0, len(predictions), 1), desc="Calculating perplexity"
        ):
            end_index = min(start_index + self.batch_size, len(predictions))
            batch_predictions = predictions[start_index:end_index]
            batch_prompts = prompts[start_index:end_index]

            self.tokenizer.pad_token = self.tokenizer.eos_token
            encoded_batch = self.tokenizer(
                batch_predictions,
                add_special_tokens=False,
                padding=True,
                truncation=False,
                return_tensors="pt",
                return_attention_mask=True,
            ).to(self.device)

            input_ids = encoded_batch["input_ids"]
            attention_mask = encoded_batch["attention_mask"]

            with torch.no_grad():
                logits = self.model(input_ids, attention_mask=attention_mask).logits

            # Filter the tokens that has a probability higher than a threshold
            logits, input_ids, attention_mask = self._filter_on_threshold(
                logits, input_ids, attention_mask
            )

            for i in range(len(batch_predictions)):
                # Calculate the length of the prompt for each example
                prompt_length = len(
                    self.tokenizer.encode(batch_prompts[i], add_special_tokens=False)
                )

                # Shift logits and labels to exclude the prompt
                shift_logits = logits[i, prompt_length:-1, :].contiguous()
                shift_labels = input_ids[i, prompt_length + 1:].contiguous()
                shift_attention_mask = attention_mask[
                                       i, prompt_length + 1:
                                       ].contiguous()

                # Ensure at least one token remains for perplexity calculation
                if shift_labels.size(0) > 0:
                    # Flatten logits and labels to match shapes
                    flat_logits = shift_logits.view(
                        -1, shift_logits.size(-1)
                    )  # (total_tokens, vocab_size)
                    flat_labels = shift_labels.view(-1)  # (total_tokens)

                    # Calculate loss for the flattened logits and labels
                    loss = torch.nn.functional.cross_entropy(
                        flat_logits, flat_labels, reduction="none"
                    )

                    # Apply attention mask to the loss
                    masked_loss = loss * shift_attention_mask.view(-1)

                    # Calculate perplexity
                    perplexity = torch.exp(
                        masked_loss.sum() / shift_attention_mask.sum()
                    )
                    ppls.append(perplexity.item())

            gc.collect()  # Collect garbage at the end of each batch
            torch.cuda.empty_cache()

            return {
                "prompts": prompts,
                "predictions": predictions,
                "perplexities": ppls,
            }

    def to_tokens_and_probs(self, input_texts: list[str], as_log_probs: bool = False) -> list[list[tuple[str, float]]]:
        input_ids = self.tokenizer(input_texts, padding=True, return_tensors="pt").input_ids
        outputs = self.model(input_ids)
        if as_log_probs:
            probs = torch.log_softmax(outputs.logits, dim=-1).detach()
        else:
            probs = torch.softmax(outputs.logits, dim=-1).detach()

        # collect the probability of the generated token -- probability at index 0 corresponds to the token at index 1
        probs = probs[:, :-1, :]
        input_ids = input_ids[:, 1:]
        gen_probs = torch.gather(probs, 2, input_ids[:, :, None]).squeeze(-1)

        batch = []
        for input_sentence, input_probs in zip(input_ids, gen_probs):
            text_sequence = []
            for token, p in zip(input_sentence, input_probs):
                if token not in self.tokenizer.all_special_ids:
                    text_sequence.append((self.tokenizer.decode(token), p.item()))
            batch.append(text_sequence)
        return batch

    def _filter_on_threshold(self, logits, input_ids, attention_mask, threshold: float) -> tuple[
        Tensor, Tensor, Tensor]:
        """
        Filter the tokens that has a probability higher than a threshold
        Args:
        input (Tensor): the input tensor
        target (Tensor): the target tensor
        Returns:
        Tuple[Tensor, Tensor]: the filtered input and target tensors
        """
        probs = torch.softmax(logits, dim=-1).detach()

        # collect the probability of the generated token -- probability at index 0 corresponds to the token at index 1
        probs = probs[:, :-1, :]
        input_ids = input_ids[:, 1:]
        logits = logits[:, :-1, :]
        attention_mask = attention_mask[:, 1:]
        gen_probs = torch.gather(probs, 2, input_ids[:, :, None]).squeeze(-1)

        filtered_input_ids = []
        filtered_logits = []
        for input_sentence, input_probs, mask, logit in zip(input_ids, gen_probs, attention_mask, logits):
            # For each input sentence get the tokens, logits filter the tokens that has a probability higher than a threshold
            input_ids_masked = input_sentence[mask == 1]
            logits_masked = logit[mask == 1]
            probs_masked = input_probs[mask == 1]

            # Filter the tokens that has a probability lower than a threshold
            filtered_input_ids.append(input_ids_masked[probs_masked < threshold])
            filtered_logits.append(logits_masked[probs_masked < threshold])

        # add the filtered_input_ids and filtered_logits to have the same shape as the input_ids and logits
        # filtered_input_ids has shape (batch_size, num_tokens) I need to add the padding tokens so that it has the same shape as input_ids
        # filtered_logits has shape (batch_size, num_tokens, vocab_size) I need to add the padding tokens so that it has the same shape as logits
        return self._pad_tensors(filtered_logits, filtered_input_ids, attention_mask)
        # max_len = max([len(input_sentence) for input_sentence in filtered_input_ids])
        # new_attention_mask = []
        # for i in range(len(filtered_input_ids)):
        #     num_pad = max_len - len(filtered_input_ids[i])
        #     new_attention_mask.append(
        #         torch.cat([torch.ones_like(filtered_input_ids[i]), torch.zeros(num_pad, dtype=torch.long)]))
        #     filtered_input_ids[i] = torch.cat([filtered_input_ids[i], torch.full((num_pad,), 0, dtype=torch.long)])
        #     filtered_logits[i] = torch.cat(
        #         [filtered_logits[i], torch.full((num_pad, logits.shape[-1]), -100, dtype=torch.float32)])
        # return torch.stack(filtered_logits), torch.stack(filtered_input_ids), torch.stack(new_attention_mask)

    def _compute(self, prompts: list[str] | None, predictions: list[str], add_start_token: bool):
        loss_fct = CrossEntropyLoss(reduction="none")
        ppls = []
        for start_index in logging.tqdm(range(0, len(predictions), self.batch_size)):
            end_index = min(start_index + self.batch_size, len(predictions))

            encodings = self.tokenizer(
                predictions[start_index:end_index],
                add_special_tokens=False,
                padding=True,
                truncation=False,
                return_tensors="pt",
                return_attention_mask=True,
            ).to(self.device)

            encoded_batch = encodings["input_ids"]
            attn_mask = encodings["attention_mask"]

            # check that each input is long enough:
            if add_start_token:
                assert torch.all(torch.ge(attn_mask.sum(1), 1)), "Each input text must be at least one token long."
            else:
                assert torch.all(
                    torch.ge(attn_mask.sum(1), 2)
                ), "When add_start_token=False, each input text must be at least two tokens long. Run with add_start_token=True if inputting strings of only one token, and remove all empty input strings."

            bos_tokens_tensor = torch.tensor([[self.tokenizer.bos_token_id]] * encoded_batch.size(dim=0)).to(
                self.device)
            encoded_batch = torch.cat([bos_tokens_tensor, encoded_batch], dim=1)
            attn_mask = torch.cat(
                [torch.ones(bos_tokens_tensor.size(), dtype=torch.int64).to(self.device), attn_mask], dim=1
            )

            labels = encoded_batch

            with torch.no_grad():
                out_logits = self.model(encoded_batch, attention_mask=attn_mask).logits

            if prompts:
                # shift the logits and labels to exclude the prompt
                shifted_logits = []
                shifted_labels = []
                shifted_attn_mask = []
                for i in range(start_index, end_index):
                    prompt_length = len(
                        self.tokenizer.encode(prompts[i], add_special_tokens=False)
                    )
                    shifted_logits.append(out_logits[i, prompt_length:attn_mask[i].sum(), :].contiguous())
                    shifted_labels.append(labels[i, prompt_length:attn_mask[i].sum()].contiguous())
                    shifted_attn_mask.append(attn_mask[i, prompt_length:attn_mask[i].sum()].contiguous())

                out_logits, labels, attn_mask = self._pad_tensors(shifted_logits, shifted_labels, shifted_attn_mask)

            # Filter the tokens that has a probability higher than a threshold
            out_logits, labels, attn_mask = self._filter_on_threshold(out_logits, labels, attn_mask, 0.99)

            perplexity_batch = torch.exp(
                (loss_fct(out_logits.transpose(1, 2), labels) * attn_mask).sum(1)
                / attn_mask.sum(1)
            )
            ppls += perplexity_batch.tolist()

        return {"perplexities": ppls, "mean_perplexity": np.mean(ppls)}


    def _pad_tensors(self, logits: list[Tensor], labels:list[Tensor], attn_mask:list[Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        max_len = max([len(label) for label in labels])
        new_attn_mask = []
        for i in range(len(labels)):
            num_pad = max_len - len(labels[i])
            new_attn_mask.append(
                torch.cat([torch.ones_like(labels[i]), torch.zeros(num_pad, dtype=torch.long)]))
            labels[i] = torch.cat([labels[i], torch.full((num_pad,), self.tokenizer.pad_token, dtype=torch.long)])
            logits[i] = torch.cat(
                [logits[i], torch.full((num_pad, logits[0].shape[-1]), 0, dtype=torch.float32)])
        return torch.stack(logits), torch.stack(labels), torch.stack(new_attn_mask)