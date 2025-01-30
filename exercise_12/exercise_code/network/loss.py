import torch
from torch.nn.functional import log_softmax
from torch import nn


class SmoothCrossEntropyLoss(nn.Module):
    def __init__(self, smoothing):
        super().__init__()
        self.smoothing = smoothing
        assert 0 <= self.smoothing < 1

    def _smooth_one_hot(self, targets: torch.Tensor, n_classes):
        """
        Transforms token ids from targets into smoothed one hot vectors of dimension n_classes (=vocab_size)
        Args:
            targets: Decoder Tokens
            n_classes: Dimension of one hot vectors (=vocab_size)
            
        Shape: (batch_size is optional)
            - targets: (batch_size, sequence_length_decoder)
            - output: (batch_size, sequence_length_decoder, n_classes)
        """

        with torch.no_grad():
            targets = torch.empty(size=tuple(targets.size()) + (n_classes,), device=targets.device).fill_(
                self.smoothing / (n_classes - 1)).scatter_(-1, targets.data.unsqueeze(-1), 1. - self.smoothing)
        return targets

    def forward(self, logits, targets, mask = None, lengths = None):
        """
        Computes Cross Entropy Loss from logits. Assumes a smoothed output probability and masks out paddings.
        Args:
            logits: Transformer output distributions over vocabulary
            targets: Target token ids
            mask: Optional Padding Mask for targets
            lengths: Optional lengths of sentences in batch to compute the loss per item
            
        Shape: (batch_size is optional)
            - logits: (batch_size, sequence_length_decoder, vocab_size)
            - targets: (batch_size, sequence_length_decoder)
            - mask: (batch_size, sequence_length_decoder)
            - lengths: (batch_size)
        """

        if lengths is None:
            lengths = torch.tensor([logits.shape[-1]])
        
        if mask is None:
            mask = torch.tensor([1])

        targets_one_hot = self._smooth_one_hot(targets, logits.shape[-1])
        loss = log_softmax(logits, -1)
        loss = (- loss * targets_one_hot * mask.unsqueeze(-1)) / lengths[..., None, None]
        return torch.sum(loss) / len(lengths)
