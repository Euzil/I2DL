from torch import nn
import torch

from ..network import MultiHeadAttention
from ..network import FeedForwardNeuralNetwork
from ..util.transformer_util import create_causal_mask

class DecoderBlock(nn.Module):

    def __init__(self,
                 d_model: int,
                 d_k: int,
                 d_v: int,
                 n_heads: int,
                 d_ff: int,
                 dropout: float = 0.0):
        """

        Args:
            d_model: Dimension of Embedding
            d_k: Dimension of Keys and Queries
            d_v: Dimension of Values
            n_heads: Number of Attention Heads
            d_ff: Dimension of hidden layer
            dropout: Dropout probability
        """
        super().__init__()

        self.causal_multi_head = None
        self.layer_norm1 = None
        self.cross_multi_head = None
        self.layer_norm2 = None
        self.ffn = None
        self.layer_norm3 = None

        ########################################################################
        # TODO:                                                                #
        #   Task 7: Initialize the Decoder Block                               #
        #            You will need:                                            #
        #                           - Causal Multi-Head Self-Attention layer   #
        #                           - Layer Normalization                      #
        #                           - Multi-Head Cross-Attention layer         #
        #                           - Layer Normalization                      #
        #                           - Feed forward neural network layer        #
        #                           - Layer Normalization                      #
        #                                                                      #
        # Hint 7: Check out the pytorch layer norm module                      #
        ########################################################################


        # Causal Multi-Head Self-Attention layer
        self.causal_multi_head = MultiHeadAttention(d_model=d_model,
                                                   d_k=d_k,
                                                   d_v=d_v,
                                                   n_heads=n_heads,
                                                   dropout=dropout)
        
        # First Layer Normalization
        self.layer_norm1 = nn.LayerNorm(d_model)
        
        # Cross-Attention layer
        self.cross_multi_head = MultiHeadAttention(d_model=d_model,
                                                  d_k=d_k,
                                                  d_v=d_v,
                                                  n_heads=n_heads,
                                                  dropout=dropout)
        
        # Second Layer Normalization
        self.layer_norm2 = nn.LayerNorm(d_model)
        
        # Feed Forward Network
        self.ffn = FeedForwardNeuralNetwork(d_model=d_model,
                                          d_ff=d_ff,
                                          dropout=dropout)
        
        # Third Layer Normalization
        self.layer_norm3 = nn.LayerNorm(d_model)

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def forward(self,
                inputs: torch.Tensor,
                context: torch.Tensor,
                causal_mask: torch.Tensor,
                pad_mask: torch.Tensor = None) -> torch.Tensor:
        """

        Args:
            inputs: Inputs from the Decoder
            context: Context from the Encoder
            causal_mask: Mask used for Causal Self Attention
            pad_mask: Optional Padding Mask used for Cross Attention

        Shape: 
            - inputs: (batch_size, sequence_length_decoder, d_model)
            - context: (batch_size, sequence_length_encoder, d_model)
            - causal_mask: (batch_size, sequence_length_decoder, sequence_length_decoder)
            - pad_mask: (batch_size, sequence_length_decoder, sequence_length_encoder)
            - outputs: (batch_size, sequence_length_decoder, d_model)
        """
        outputs = None

        ########################################################################
        # TODO:                                                                #
        #   Task 7: Implement the forward pass of the decoder block            #
        #   Task 10: Pass on the padding mask                                  #
        #                                                                      #
        # Hint 7:                                                              #
        #       - Don't forget the residual connections!                       #
        #       - Remember where we need the causal mask, forget about the     #
        #         other mask for now!                                          #
        # Hints 10:                                                            #
        #       - We have already combined the causal_mask with the pad_mask   #
        #         for you, all you have to do is pass it on to the "other"     #
        #         module                                                       #
        ########################################################################


        # Causal Self-Attention with residual connection
        causal_output = self.causal_multi_head(inputs, inputs, inputs, causal_mask)
        outputs = self.layer_norm1(inputs + causal_output)
        
        # Cross-Attention with residual connection
        cross_output = self.cross_multi_head(outputs, context, context, pad_mask)
        outputs = self.layer_norm2(outputs + cross_output)
        
        # Feed Forward Network with residual connection
        ffn_output = self.ffn(outputs)
        outputs = self.layer_norm3(outputs + ffn_output)

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

        return outputs