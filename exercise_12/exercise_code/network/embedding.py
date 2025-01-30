from torch import nn
import torch

def positional_encoding(d_model: int,
                        max_length: int) -> torch.Tensor:
    """
    Computes the positional encoding matrix
    Args:
        d_model: Dimension of Embedding
        max_length: Maximums sequence length

    Shape:
        - output: (max_length, d_model)
    """

    i = torch.arange(0, d_model, 2) / d_model
    pos = torch.arange(0, max_length)[:, None]
    
    angle_freq = torch.exp(i * (-torch.log(torch.Tensor([10000]))))

    output = torch.zeros((max_length, d_model))

    output[:, 0::2] = torch.sin(pos * angle_freq)
    output[:, 1::2] = torch.cos(pos * angle_freq)

    return output


class Embedding(nn.Module):

    def __init__(self,
                 vocab_size: int,
                 d_model: int,
                 max_length: int,
                 dropout: float = 0.0):
        """

        Args:
            vocab_size: Number of elements in the vocabulary
            d_model: Dimension of Embedding
            max_length: Maximum sequence length
        """
        super().__init__()

        self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                      embedding_dim=d_model)

        self.pos_encoding = nn.Parameter(data=positional_encoding(d_model=d_model, max_length=max_length),
                                         requires_grad=False)
        
        self.dropout = None

        ########################################################################
        # TODO:                                                                #
        #   Task 11: Initialize the dropout layer (torch.nn implementation)    #
        #                                                                      #           
        ########################################################################


        pass

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################


    def forward(self,
                inputs: torch.Tensor) -> torch.Tensor:
        """
        The forward function takes in tensors of token ids and transforms them into vector embeddings. 
        It then adds the positional encoding to the embeddings, and if configured, performs dropout on the layer!

        Args:
            inputs: Batched Sequence of Token Ids

        Shape:
            - inputs: (batch_size, sequence_length)
            - outputs: (batch_size, sequence_length, d_model)
        """

        sequence_length = inputs.shape[-1]
        outputs = self.embedding(inputs) + self.pos_encoding[:sequence_length]

        ########################################################################
        # TODO:                                                                #
        #   Task 11:                                                           #
        #       - Add dropout to the outputs                                   #
        ########################################################################


        pass

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

        return outputs