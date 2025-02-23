# Custom torch cross Entropy function that calculates cross entropy on the tokens that has a probability lower than a threshold
import torch
from torch import Tensor
from torch.nn import CrossEntropyLoss

class CrossEntropyWithThreshold(CrossEntropyLoss):
    def __init__(
            self,
            threshold: float = 0.99,
            weight: Tensor | None = None,
            size_average=None,
            ignore_index: int = -100,
            reduce=None,
            reduction: str = "mean",
            label_smoothing: float = 0.0,
    ) -> None:
        """
        Custom cross entropy function that calculates cross entropy on the tokens that has a probability lower than a threshold
        Args:
        threshold (float): the threshold to filter the tokens
        weight (Tensor, optional): a manual rescaling weight given to each class.
            If given, has to be a Tensor of size `C` and floating point dtype
        size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
            the losses are averaged over each loss element in the batch. Note that for
            some losses, there are multiple elements per sample. If the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch. Ignored
            when :attr:`reduce` is ``False``. Default: ``True``
        ignore_index (int, optional): Specifies a target value that is ignored
            and does not contribute to the input gradient. When :attr:`size_average` is
            ``True``, the loss is averaged over non-ignored targets. Note that
            :attr:`ignore_index` is only applicable when the target contains class indices.
        reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
            losses are averaged or summed over observations for each minibatch depending
            on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
            batch element instead and ignores :attr:`size_average`. Default: ``True``
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will
            be applied, ``'mean'``: the weighted mean of the output is taken,
            ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in
            the meantime, specifying either of those two args will override
            :attr:`reduction`. Default: ``'mean'``
        label_smoothing (float, optional): A float in [0.0, 1.0]. Specifies the amount
            of smoothing when computing the loss, where 0.0 means no smoothing. The targets
            become a mixture of the original ground truth and a uniform distribution as described in
            `Rethinking the Inception Architecture for Computer Vision <https://arxiv.org/abs/1512.00567>`__. Default: :math:`0.0`.
        """
        super().__init__(
            weight=weight,
            size_average=size_average,
            ignore_index=ignore_index,
            reduce=reduce,
            reduction=reduction,
            label_smoothing=label_smoothing,
        )
        self.threshold = threshold

    def forward(self, input: Tensor, target: Tensor, attention_mask:Tensor) -> Tensor:
        """
        Calculate the cross entropy loss on the tokens that has a probability lower than a threshold
        Args:
        input (Tensor): the input tensor
        target (Tensor): the target tensor
        Returns:
        Tensor: the cross entropy loss
        """
        input, target = self._filter_on_threshold(input, target, attention_mask)
        return super().forward(input, target)


    def _filter_on_threshold(self, logits, input_ids, attention_mask):
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
        logits = logits[:, 1:, :]
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
            filtered_input_ids.append(input_ids_masked[probs_masked < self.threshold])
            filtered_logits.append(logits_masked[probs_masked < self.threshold])

        # add the filtered_input_ids and filtered_logits to have the same shape as the input_ids and logits
        # filtered_input_ids has shape (batch_size, num_tokens) I need to add the padding tokens so that it has the same shape as input_ids
        # filtered_logits has shape (batch_size, num_tokens, vocab_size) I need to add the padding tokens so that it has the same shape as logits
        max_len = max([len(input_sentence) for input_sentence in filtered_input_ids])
        new_attention_mask = []
        for i in range(len(filtered_input_ids)):
            num_pad = max_len - len(filtered_input_ids[i])
            new_attention_mask.append(torch.cat([torch.ones_like(filtered_input_ids[i]), torch.zeros(num_pad, dtype=torch.long)]))
            filtered_input_ids[i] = torch.cat([filtered_input_ids[i], torch.full((num_pad,), 0, dtype=torch.long)])
            filtered_logits[i] = torch.cat([filtered_logits[i], torch.full((num_pad, logits.shape[-1]), -100, dtype=torch.float32)])
        return torch.stack(filtered_logits), torch.stack(filtered_input_ids), torch.stack(new_attention_mask)


