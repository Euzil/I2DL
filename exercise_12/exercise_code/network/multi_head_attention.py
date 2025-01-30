from torch import nn
import torch    

from ..network import ScaledDotAttention

class MultiHeadAttention(nn.Module):

    def __init__(self,
                 d_model: int,
                 d_k: int,
                 d_v: int,
                 n_heads: int,
                 dropout: float = 0.0):
        """

        Args:
            d_model: Dimension of Embedding
            d_k: Dimension of Keys and Queries
            d_v: Dimension of Values
            n_heads: Number of Attention Heads
            dropout: Dropout probability
        """
        super().__init__()

        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v

        self.weights_q = nn.Linear(in_features=d_model, out_features=n_heads * d_k, bias=False)
        self.weights_k = nn.Linear(in_features=d_model, out_features=n_heads * d_k, bias=False)
        self.weights_v = nn.Linear(in_features=d_model, out_features=n_heads * d_v, bias=False)

        self.attention = ScaledDotAttention(d_k=d_k, dropout=dropout)

        self.project = nn.Linear(in_features=n_heads * d_v, out_features=d_model, bias=False)

        self.dropout = None

        ########################################################################
        # TODO:                                                                #
        #  Task 11:                                                            #
        #       -Initialize the dropout layer (torch.nn implementation)        #
        #                                                                      #
        ########################################################################


        self.dropout = nn.Dropout(p=dropout)

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################



    def forward(self,
                q: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        """

        Args:
            q: Query Inputs
            k: Key Inputs
            v: Value Inputs
            mask: Optional Causal or Padding Mask

        Shape:
            - q: (batch_size, sequence_length_queries, d_model)
            - k: (batch_size, sequence_length_keys, d_model)
            - v: (batch_size, sequence_length_keys, d_model)
            - mask: (batch_size, sequence_length_queries, sequence_length_keys)
            - outputs: (batch_size, sequence_length_queries, d_model)
        """

        batch_size, sequence_length_queries, _ = q.size()
        _, sequence_length_keys, _ = k.size()

        q = self.weights_q(q)
        k = self.weights_k(k)
        v = self.weights_v(v)

        q = q.reshape(batch_size, sequence_length_queries, self.n_heads, self.d_k)
        q = q.transpose(-3, -2)

        k = k.reshape(batch_size, sequence_length_keys, self.n_heads, self.d_k)
        k = k.transpose(-3, -2)

        v = v.reshape(batch_size, sequence_length_keys, self.n_heads, self.d_v)
        v = v.transpose(-3, -2)

        ########################################################################
        # TODO:                                                                #
        #   Task 6:                                                            #
        #       - If a mask is given, add an empty dimension at dim=1          #
        #       - Pass the mask to the ScaledDotAttention layer                #
        #                                                                      #
        # Hints 6:                                                             #
        #       - Use unsqueeze() to add dimensions at the correct location    #
        ########################################################################


        # Handle mask for multiple heads (Task 6)
        if mask is not None:
            # Add dimension for n_heads
            mask = mask.unsqueeze(1)

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

        outputs = self.attention(q, k, v, mask)

        outputs = outputs.transpose(-3, -2)
        outputs = outputs.reshape(batch_size, sequence_length_queries, self.n_heads * self.d_v)

        outputs = self.project(outputs)

        ########################################################################
        # TODO:                                                                #
        #  Task 11:                                                            #
        #       - Add dropout as a final step after the projection layer       #
        #                                                                      #
        ########################################################################


        # Apply dropout after projection (Task 11)
        outputs = self.dropout(outputs)

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

        return outputs
    