from torch import nn
import torch

from ..network import DecoderBlock
from ..util.transformer_util import create_causal_mask


class Decoder(nn.Module):

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
            n: Number of Decoder Blocks
            dropout: Dropout probability
        """
        super().__init__()

        self.stack = nn.ModuleList([DecoderBlock(d_model=d_model,
                                                 d_k=d_k,
                                                 d_v=d_v,
                                                 n_heads=n_heads,
                                                 d_ff=d_ff,
                                                 dropout=dropout) for _ in range(n)])
        

    def forward(self,
                inputs: torch.Tensor,
                context: torch.Tensor,
                decoder_mask: torch.Tensor = None,
                encoder_mask: torch.Tensor = None) -> torch.Tensor:
        """

        Args:
            inputs: Inputs from the Decoder
            context: Context from the Encoder
            decoder_mask: Optional Padding Mask for Decoder Inputs
            encoder_mask: Optional Padding Mask for Encoder Inputs

        Shape: 
            - inputs: (batch_size, sequence_length_decoder, d_model)
            - context: (batch_size, sequence_length_encoder, d_model)
            - decoder_mask: (batch_size, sequence_length_decoder, sequence_length_decoder)
            - encoder_mask: (batch_size, sequence_length_encoder, sequence_length_encoder)
            - outputs: (batch_size, sequence_length_decoder, d_model)
        """

        # Create a causal mask for the decoder
        causal_mask = create_causal_mask(inputs.shape[-2]).to(inputs.device)

        # Combine the causal mask with the decoder mask - We haven't discussed this yet so don't worry about it!
        if decoder_mask is not None:
            causal_mask = causal_mask * decoder_mask

        # This is just so we can loop through the decoder blocks nicer - it is completely unnecessary!
        outputs = inputs

        # Loop through the decoder blocks
        for decoder in self.stack:
            outputs = decoder(outputs, context, causal_mask, encoder_mask)

        return outputs