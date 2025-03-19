import numpy as np
from torch import Tensor
import torch
from torch.nn import CrossEntropyLoss
import gc
from evaluate import logging
import traceback


class Perplexity:
    def __init__(
        self,
        model,
        tokenizer,
        device: str | None = None,
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
                "mps",
            ], "device should be either gpu or cpu."
            if device == "gpu":
                device = "cuda"
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.batch_size = batch_size
        self.loss_fct = CrossEntropyLoss(reduction="none")

    def compute(
        self,
        prompts: list[str] | None,
        predictions: list[str],
        add_start_token: bool = True,
        thresholds: list | None = None,
    ) -> dict:
        """
        Compute the perplexity of the generated text if the prompts of the generated text are given the perplexity computation is done on the generated text
        Else the perplexity computation is done on the prompt + generated text

        :param prompts: prompts used to generate the text if provided the perplexity is computed on the generated text else the perplexity is computed on the prompt + generated text
        :param predictions: predictions as prompts + generated text to condition on the prompts
        :param add_start_token: if True add the start token to the input text
        :param thresholds: list of thresholds to filter the tokens
        :return: dictionary of results
        """
        if thresholds is None:
            thresholds = [1.01]

        loss_fct = CrossEntropyLoss(reduction="none")
        col = {}
        for val in thresholds:
            col[str(val)] = {
                "total_tokens": [],
                "filtered_tokens": [],
                "ppls": [],
                "longest_sequences": [],
                "sample_probs": [],
            }

        for start_index in logging.tqdm(range(0, len(predictions), self.batch_size)):
            print(f"Index: {start_index}")
            try:
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
                    bos_tokens_tensor = torch.tensor([[self.tokenizer.bos_token_id]] * encoded_batch.size(dim=0)).to(
                        self.device
                    )
                    encoded_batch = torch.cat([bos_tokens_tensor, encoded_batch], dim=1)
                    attn_mask = torch.cat(
                        [
                            torch.ones(bos_tokens_tensor.size(), dtype=torch.int64).to(self.device),
                            attn_mask,
                        ],
                        dim=1,
                    )
                else:
                    assert torch.all(
                        torch.ge(attn_mask.sum(1), 2)
                    ), "When add_start_token=False, each input text must be at least two tokens long. Run with add_start_token=True if inputting strings of only one token, and remove all empty input strings."

                labels = encoded_batch

                with torch.no_grad():
                    out_logits = self.model(encoded_batch, attention_mask=attn_mask).logits

                logging.DEBUG("shape of encoded_batch: ", encoded_batch.shape)
                logging.DEBUG("shape of attn_mask: ", attn_mask.shape)

                if prompts:
                    # shift the logits and labels to exclude the prompt
                    shifted_logits = []
                    shifted_labels = []
                    shifted_attn_mask = []
                    for i in range(self.batch_size):
                        prompt_length = len(self.tokenizer.encode(prompts[i], add_special_tokens=False))
                        shifted_logits.append(out_logits[i, prompt_length : attn_mask[i].sum(), :].contiguous())
                        shifted_labels.append(labels[i, prompt_length : attn_mask[i].sum()].contiguous())
                        shifted_attn_mask.append(attn_mask[i, prompt_length : attn_mask[i].sum()].contiguous())

                    out_logits, labels, attn_mask = self._pad_tensors(shifted_logits, shifted_labels, shifted_attn_mask)

                    logging.DEBUG("shape of out_logits after shifting: ", out_logits.shape)
                    logging.DEBUG("shape of labels after shifting: ", labels.shape)

                # Filter the tokens that has a probability higher than a threshold
                for idx, val in enumerate(thresholds):
                    (temp_out_logits, temp_labels, temp_attn_mask), generated_probs, tt, ft, ls, gp = (
                        self._filter_on_threshold(out_logits, labels, attn_mask, val)
                    )
                    col[str(val)]["total_tokens"] += tt
                    col[str(val)]["filtered_tokens"] += ft

                    perplexity_batch = torch.exp(
                        (loss_fct(temp_out_logits.transpose(1, 2), temp_labels) * temp_attn_mask).sum(1)
                        / temp_attn_mask.sum(1)
                    )
                    col[str(val)]["ppls"] += perplexity_batch.tolist()
                    col[str(val)]["longest_sequences"] += ls
                    if idx == 0:
                        col[str(val)]["sample_probs"] += gp

                # Collect garbage at the end of each batch
                gc.collect()
                torch.cuda.empty_cache()
            except RuntimeError as e:
                print(f"Error: {e}")
                print(f"Traceback: {traceback.format_exc()}")
                for idx, val in enumerate(thresholds):
                    col[str(val)]["ppls"] += [np.nanmean(col[str(val)]["ppls"])] * self.batch_size
                    col[str(val)]["longest_sequences"] += [
                        np.nanmean(col[str(val)]["longest_sequences"])
                    ] * self.batch_size
                    col[str(val)]["sample_probs"] += [[0]] * self.batch_size

        results = {}
        for val in thresholds:
            results[str(val)] = {
                "mean_perplexity": np.nanmean(col[str(val)]["ppls"]),
                "perplexities": col[str(val)]["ppls"],
                "filtered_token_percentage": sum(col[str(val)]["filtered_tokens"])
                / (sum(col[str(val)]["total_tokens"]) + 1e-9),
                "longest_filtered_sequences": col[str(val)]["longest_sequences"],
                "sample_probs": col[str(val)]["sample_probs"],
            }

        return results

    def _filter_on_threshold(
        self,
        logits: Tensor,  # (batch_size, num_tokens, vocab_size)
        input_ids: Tensor,  # (batch_size, num_tokens)
        attention_mask: Tensor,  # (batch_size, num_tokens)
        threshold: float,
    ) -> tuple[
        tuple[Tensor, Tensor, Tensor],
        list[float],
        list[int],
        list[int],
        list[float],
        list[float],
    ]:
        """
        Filter the tokens that has a probability higher than a threshold
        :param logits: logits
        :param input_ids: input_ids
        :param attention_mask: attention_mask
        :param threshold: threshold to filter the tokens
        :return: tuple of filtered logits, filtered input_ids, filtered attention_mask, list of token probabilities, total tokens, filtered tokens, list of longest filtered sequence
        """
        probs_collection = []
        token_count = []
        filter_count = []
        longest_filtered_sequences = []

        probs = torch.softmax(logits, dim=-1).detach()

        # collect the probability of the generated token -- probability at index 0 corresponds to the token at index 1
        probs = probs[:, :-1, :]
        input_ids = input_ids[:, 1:]
        logits = logits[:, :-1, :]
        attention_mask = attention_mask[:, 1:]
        gen_probs = torch.gather(probs, 2, input_ids[:, :, None]).squeeze(-1)

        filtered_input_ids = []
        filtered_logits = []
        filtered_attention_mask = []
        for input_sentence, input_probs, mask, logit in zip(input_ids, gen_probs, attention_mask, logits):
            # For each input sentence get the tokens, logits filter the tokens that has a probability higher than a threshold
            input_ids_masked = input_sentence[mask == 1]
            logits_masked = logit[mask == 1]
            probs_masked = input_probs[mask == 1]
            mask = mask[mask == 1]

            # Filter the tokens that has a probability lower than a threshold
            threshold_mask = probs_masked < threshold
            filtered_input_ids.append(input_ids_masked[threshold_mask])
            filtered_logits.append(logits_masked[threshold_mask])
            filtered_attention_mask.append(mask[threshold_mask])

            # collect the probabilities of the filtered tokens
            probs_collection += probs_masked.tolist()
            token_count.append(len(input_ids_masked))
            filter_count.append(len(input_ids_masked) - len(input_ids_masked[threshold_mask]))
            # count the longest sequence of filtered tokens
            threshold_mask_str = "".join(map(str, threshold_mask.int().tolist()))
            longest_filtered_sequences.append(
                max(map(len, threshold_mask_str.split("1")))
                / (len(threshold_mask_str) + (1e-9 if len(threshold_mask_str) == 0 else 0))
            )

        # filtered_input_ids has shape (batch_size, num_tokens), add the padding tokens so that it has the same shape as input_ids
        # filtered_logits has shape (batch_size, num_tokens, vocab_size), add the padding tokens so that it has the same shape as logits
        return (
            self._pad_tensors(filtered_logits, filtered_input_ids, filtered_attention_mask),
            probs_collection,
            token_count,
            filter_count,
            longest_filtered_sequences,
            gen_probs.tolist(),
        )

    def _pad_tensors(
        self, logits: list[Tensor], labels: list[Tensor], attn_mask: list[Tensor]
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Pad the logits and labels to have the same shape after filtering
        :param logits: list of logits
        :param labels: list of labels
        :param attn_mask:  list of attention masks
        :return: padded logits, padded labels, padded attention masks as tuple of tensors
        """
        max_len = max([len(label) for label in labels])
        new_attn_mask = []
        for i in range(len(labels)):
            num_pad = max_len - attn_mask[i].sum().detach().item()
            new_attn_mask.append(
                torch.cat(
                    [
                        torch.ones_like(attn_mask[i][attn_mask[i] == 1]),
                        torch.zeros(num_pad, dtype=torch.long).to(self.device),
                    ]
                )
            )
            labels[i] = torch.cat(
                [
                    labels[i][attn_mask[i] == 1],
                    torch.full((num_pad,), self.tokenizer.pad_token_id, dtype=torch.long).to(self.device),
                ]
            )
            logits[i] = torch.cat(
                [
                    logits[i][attn_mask[i] == 1, :],
                    torch.full((num_pad, logits[0].shape[-1]), 0, dtype=torch.float32).to(self.device),
                ]
            )
        return torch.stack(logits), torch.stack(labels), torch.stack(new_attn_mask)
