import numpy as np


class Sigmoid:
    def __init__(self):
        pass

    def forward(self, x):
        """
        :param x: Inputs, of any shape.

        :return out: Outputs, of the same shape as x.
        :return cache: Cache, stored for backward computation, of the same shape as x.
        """
        shape = x.shape
        out, cache = np.zeros(shape), np.zeros(shape)
        ########################################################################
        # TODO:                                                                #
        # Implement the forward pass of Sigmoid activation function            #
        ########################################################################

        # 计算 sigmoid: 1/(1 + e^(-x))
        out = 1 / (1 + np.exp(-x))
        
        # 把输出保存到 cache 中，供反向传播使用
        cache = out
        
        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return out, cache

    def backward(self, dout, cache):
        """
        :param dout: Upstream gradient from the computational graph, from the Loss function
                    and up to this layer. Has the shape of the output of forward().
        :param cache: The values that were stored during forward() to the memory,
                    to be used during the backpropogation.
        :return: dx: the gradient w.r.t. input X, of the same shape as X
        """
        dx = None
        ########################################################################
        # TODO:                                                                #
        # Implement the backward pass of Sigmoid activation function           #
        ########################################################################
        

         # 从 cache 中获取前向传播的输出
        sigmoid_out = cache
        
        # 计算梯度：σ'(x) = σ(x)(1 - σ(x))
        dx = dout * sigmoid_out * (1 - sigmoid_out)

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return dx


class Relu:
    def __init__(self):
        pass

    def forward(self, x):
        """
        :param x: Inputs, of any shape.

        :return outputs: Outputs, of the same shape as x.
        :return cache: Cache, stored for backward computation, of the same shape as x.
        """
        out = None
        cache = None
        ########################################################################
        # TODO:                                                                #
        # Implement the forward pass of Relu activation function               #
        ########################################################################

         # ReLU forward: max(0,x)
        out = np.maximum(0, x)
        
        # 存储输入x用于反向传播
        cache = x

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return out, cache

    def backward(self, dout, cache):
        """
        :param dout: Upstream gradient from the computational graph, from the Loss function
                    and up to this layer. Has the shape of the output of forward().
        :param cache: The values that were stored during forward() to the memory,
                    to be used during the backpropogation.
        :return: dx: the gradient w.r.t. input X, of the same shape as X
        """
        dx = None
        ########################################################################
        # TODO:                                                                #
        # Implement the backward pass of Relu activation function              #
        ########################################################################

        # 从cache获取输入x
        x = cache
        
        # ReLU的导数：若x>0，导数为1；若x<0，导数为0
        dx = dout * (x > 0)

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return dx


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.
    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.
    Inputs:
    :param x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    :param w: A numpy array of weights, of shape (D, M)
    :param b: A numpy array of biases, of shape (M,)
    :return out: output, of shape (N, M)
    :return cache: (x, w, b)
    """
    N, M = x.shape[0], b.shape[0]
    out = np.zeros((N,M))
    ########################################################################
    # TODO: Implement the affine forward pass. Store the result in out.    #
    # You will need to reshape the input into rows.                        #
    ########################################################################

    # 将输入x重塑为(N, D)的矩阵
    x_reshaped = x.reshape(N, -1)
    
    # 计算前向传播: out = x * w + b
    out = np.dot(x_reshaped, w) + b

    ########################################################################
    #                           END OF YOUR CODE                           #
    ########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.
    Inputs:
    :param dout: Upstream derivative, of shape (N, M)
    :param cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: A numpy array of biases, of shape (M,
    :return dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    :return dw: Gradient with respect to w, of shape (D, M)
    :return db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ########################################################################
    # TODO: Implement the affine backward pass.                            #
    ########################################################################


     # 获取输入x的形状，用于后面重塑dx
    x_shape = x.shape
    
    # 将x重塑为二维矩阵，用于计算梯度
    x_reshaped = x.reshape(x.shape[0], -1)
    
    # 计算各个参数的梯度
    dx = np.dot(dout, w.T).reshape(x_shape)  # 重塑回原始输入形状
    dw = np.dot(x_reshaped.T, dout)
    db = np.sum(dout, axis=0)

    ########################################################################
    #                           END OF YOUR CODE                           #
    ########################################################################
    return dx, dw, db