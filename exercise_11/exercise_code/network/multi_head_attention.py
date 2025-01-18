from torch import nn
import torch    

from ..network import ScaledDotAttention

class MultiHeadAttention(nn.Module):

    def __init__(self,
                 d_model: int,
                 d_k: int,
                 d_v: int,
                 n_heads: int):
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

        self.weights_q = None
        self.weights_k = None
        self.weights_v = None
        self.attention = None
        self.project = None

        ########################################################################
        # TODO:                                                                #
        #   Task 3:                                                            #
        #       -Initialize all weight layers as linear layers                 #
        #       -Initialize the ScaledDotAttention                             #
        #       -Initialize the projection layer as a linear layer             #
        #                                                                      #
        # Hints 3:                                                             #
        #       - Instead of initializing several weight layers for each head, #
        #         you can create one large weight matrix. This speed up        #
        #         the forward pass, since we dont have to loop through all     #
        #         heads!                                                       #
        #       - All linear layers should only be a weight without a bias!    #
        ########################################################################


        # 初始化查询、键、值的权重矩阵（无偏置）
        self.weights_q = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.weights_k = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.weights_v = nn.Linear(d_model, n_heads * d_v, bias=False)
        
        # 初始化注意力层
        self.attention = ScaledDotAttention(d_k)
        
        # 初始化投影层（无偏置）
        self.project = nn.Linear(n_heads * d_v, d_model, bias=False)

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def forward(self,
                q: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor) -> torch.Tensor:
        """

        Args:
            q: Query Inputs
            k: Key Inputs
            v: Value Inputs

        Shape:
            - q: (batch_size, sequence_length_queries, d_model)
            - k: (batch_size, sequence_length_keys, d_model)
            - v: (batch_size, sequence_length_keys, d_model)
            - outputs: (batch_size, sequence_length_queries, d_model)
        """

        # You will need these here!
        batch_size, sequence_length_queries, _ = q.size()
        _, sequence_length_keys, _ = k.size()

        outputs = None

        ########################################################################
        # TODO:                                                                #
        #   Task 3:                                                            #
        #       - Pass q,k and v through the linear layer                      #
        #       - Split the last dimensions into n_heads and d_k od d_v        #
        #       - Swap the dimensions so that the shape matches the required   #
        #         input shapes of the ScaledDotAttention layer                 #
        #       - Pass them through the ScaledDotAttention layer               #
        #       - Swap the dimensions of the output back                       #
        #       - Combine the last two dimensions again                        #
        #       - Pass the outputs through the projection layer                #
        #                                                                      #
        # Hints 3:                                                             #
        #       - It helps to write down which dimensions you want to have on  #
        #         paper!                                                       #
        #       - Above the todo, we have already extracted the batch_size and #
        #         the sequence lengths for you!                                #
        #       - Use reshape() to split or combine dimensions                 #
        #       - Use transpose() again to swap dimensions                     #                            
        ########################################################################


        # 1. 通过线性层
        q = self.weights_q(q)  # (batch_size, seq_len_q, n_heads * d_k)
        k = self.weights_k(k)  # (batch_size, seq_len_k, n_heads * d_k)
        v = self.weights_v(v)  # (batch_size, seq_len_v, n_heads * d_v)
        
        # 2. 分割最后一个维度为 n_heads 和 d_k/d_v
        q = q.reshape(batch_size, sequence_length_queries, self.n_heads, self.d_k)
        k = k.reshape(batch_size, sequence_length_keys, self.n_heads, self.d_k)
        v = v.reshape(batch_size, sequence_length_keys, self.n_heads, self.d_v)
        
        # 3. 调整维度顺序为 (batch_size, n_heads, seq_len, d_k/d_v)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # 4. 应用注意力机制
        attention_output = self.attention(q, k, v)  # (batch_size, n_heads, seq_len_q, d_v)
        
        # 5. 调整维度顺序回来
        attention_output = attention_output.transpose(1, 2)  # (batch_size, seq_len_q, n_heads, d_v)
        
        # 6. 合并最后两个维度
        outputs = attention_output.reshape(batch_size, sequence_length_queries, self.n_heads * self.d_v)
        
        # 7. 通过投影层
        outputs = self.project(outputs)

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

        return outputs
    