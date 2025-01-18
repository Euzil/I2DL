import torch


def create_causal_mask(decoder_length: int) -> torch.Tensor:
    """
    Creates a lower triangle boolean mask for decoder self attention.
    Args:
        decoder_length: Sequence length of decoder

    Shape:
        - output: (batch_size, sequence_length, sequence_length)
    """
    output = torch.ones((decoder_length, decoder_length))
    output = torch.tril(output, diagonal=0).bool()

    return output.unsqueeze_(0)
