from torch import nn
import torch
from ..network import EncoderBlock

class Encoder(nn.Module):

    def __init__(self,
                 d_model: int,
                 d_k: int,
                 d_v: int,
                 n_heads: int,
                 d_ff: int,
                 n: int,
                 dropout: float = 0.0):
        """

        Args:
            d_model: Dimension of Embedding
            d_k: Dimension of Keys and Queries
            d_v: Dimension of Values
            n_heads: Number of Attention Heads
            d_ff: Dimension of hidden layer
            n: Number of Encoder Blocks
            dropout: Dropout probability
        """
        super().__init__()

        self.stack = nn.ModuleList([EncoderBlock(d_model=d_model,
                                                 d_k=d_k,
                                                 d_v=d_v,
                                                 n_heads=n_heads,
                                                 d_ff=d_ff,
                                                 dropout=dropout) for _ in range(n)])


    def forward(self,
                inputs: torch.Tensor,
                encoder_mask: torch.Tensor = None) -> torch.Tensor:
        """

        Args:
            inputs: Inputs to the Encoder Stack
            encoder_mask: Optional Padding Mask for Encoder Inputs

        Shape:
            - inputs: (batch_size, sequence_length, d_model)
            - encoder_mask: (batch_size, 1, sequence_length)
            - outputs: (batch_size, sequence_length, d_model)
        """

        # This is just so we can loop through the encoder blocks nicer - it is completely unnecessary!
        outputs = inputs

        # Loop through the encoder blocks
        for encoder in self.stack:
            outputs = encoder(outputs, encoder_mask)

        return outputs