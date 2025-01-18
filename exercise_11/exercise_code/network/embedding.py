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

    output = None

    ########################################################################
    # TODO:                                                                #
    #   Task 4: Initialize the positional encoding layer.                  #
    #                                                                      #
    # Hints 4:                                                             #
    #       - You can copy the implementation from the notebook, just      #
    #         make sure to use torch instead of numpy!                     #
    #       - Use torch.log(torch.Tensor([10000])), to make use of the     #
    #         torch implementation of the natural logarithm.               #
    #       - Implement the alternating sin and cos functions the way we   #
    #         did in the notebook.                                         #
    ########################################################################


    # Create position encodings
    positions = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
    
    # Calculate divisions for the frequencies
    div = torch.pow(10000.0, torch.arange(0, d_model, 2, dtype=torch.float) / d_model)
    
    # Initialize output tensor
    output = torch.zeros(max_length, d_model)
    
    # Apply sin to even indices
    arguments = positions / div
    output[:, 0::2] = torch.sin(arguments)
    
    # Apply cos to odd indices
    if d_model % 2 == 1:
        # If d_model is odd, we need one less cosine term
        output[:, 1::2] = torch.cos(arguments[:, :-1])
    else:
        output[:, 1::2] = torch.cos(arguments)

    ########################################################################
    #                           END OF YOUR CODE                           #
    ########################################################################

    return output

class Embedding(nn.Module):

    def __init__(self,
                 vocab_size: int,
                 d_model: int,
                 max_length: int):
        """

        Args:
            vocab_size: Number of elements in the vocabulary
            d_model: Dimension of Embedding
            max_length: Maximum sequence length
        """
        super().__init__()

        self.embedding = None
        self.pos_encoding = None

        ########################################################################
        # TODO:                                                                #
        #   Task 1: Initialize the embedding layer (torch.nn implementation)   #
        #   Task 4: Initialize the positional encoding layer.                  #
        #                                                                      #
        # Hints 1:                                                             #
        #       - Have a look at pytorch embedding module                      #
        # Hints 4:                                                             #
        #       - Initialize it using d_model and max_length                   #
        ########################################################################


         # Initialize the embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Initialize positional encoding
        self.pos_encoding = positional_encoding(d_model, max_length)

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

        # We will convert it into a torch parameter module for you! You can treat it like a normal tensor though!
        if self.pos_encoding is not None:
            self.pos_encoding = nn.Parameter(data=self.pos_encoding, requires_grad=False)

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

        outputs = None

        # Use fancy indexing to extract the positional encodings until position sequence_length
        sequence_length = inputs.shape[-1]
        pos_encoding = 0
        if self.pos_encoding is not None:
            pos_encoding = self.pos_encoding[:sequence_length]

        ########################################################################
        # TODO:                                                                #
        #   Task 1: Compute the outputs of the embedding layer                 #
        #   Task 4: Add the positional encoding to the output                  #
        #                                                                      #
        # Hint 4: We have already extracted them for you, all you have to do   #
        #         is add them to the embeddings!                               #
        ########################################################################


        # Get embeddings from the embedding layer
        outputs = self.embedding(inputs)
        
        # Add positional encoding
        sequence_length = inputs.shape[-1]
        if self.pos_encoding is not None:
            pos_encoding = self.pos_encoding[:sequence_length]
            outputs = outputs + pos_encoding

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

        return outputs