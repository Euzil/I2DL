
import os
import pickle
import numpy as np

from exercise_code.networks.base_networks import Network


class Classifier(Network):
    """
    Classifier of the form y = sigmoid(X * W)
    """

    def __init__(self, num_features=2):
        super().__init__("classifier")

        self.num_features = num_features 
        self.W = None
        self.cache = None
        
        self.initialize_weights()

    def initialize_weights(self, weights=None):
        """
        Initialize the weight matrix W

        :param weights: optional weights for initialization
        """
        if weights is not None:
            assert weights.shape == (self.num_features + 1, 1), \
                "weights for initialization are not in the correct shape (num_features + 1, 1)"
            self.W = weights
        else:
            self.W = 0.001 * np.random.randn(self.num_features + 1, 1)

    def forward(self, X):
        """
        Performs the forward pass of the model.

        :param X: N x D array of training data. Each row is a D-dimensional point. 
            Note that it is changed to N x (D + 1) to include the bias term.
        :return: Predicted logits for the data in X, shape N x 1
                 1-dimensional array of length N with classification scores.
                 
        Note: This simple neural-network contains TWO consecutive layers: 
        A fully-connected layer and a sigmoid layer.
        """
        assert self.W is not None, "weight matrix W is not initialized"
        # add a column of 1s to the data for the bias term
        batch_size, _ = X.shape
        X = np.concatenate((X, np.ones((batch_size, 1))), axis=1)
        


        ########################################################################
        # TODO:                                                                #
        # Implement the forward pass and return the output of the model. Note  #
        # that you need to implement the function self.sigmoid() for that.     #
        # Also, save in self.cache an array of all the relevant variables that #
        # you will need to use in the backward() function. E.g.: (X, ...)      #
        #                                                                      #
        # Hint:                                                                #
        #       The function is made up of TWO functions: Affine and sigmoid.  #
        #       The sigmoid is applied to the result of the affine step.       #
        ########################################################################


        # Compute affine transformation
        Z = np.dot(X, self.W)
        
        # Apply the sigmoid function
        output = self.sigmoid(Z)
        
        # Store relevant variables for backpropagation
        self.cache = (X, Z)

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

        return output 

    def backward(self, dout):
        """
        Performs the backward pass of the model.

        :param dout: N x M array. Upsteam derivative. It is as the same shape of the forward() output.
                     If the output of forward() is z, then it is dL/dz, where L is the loss function.
        :return: dW --> Gradient of the weight matrix, w.r.t the upstream gradient 'dout'. (dL/dw)
        
        Note: Pay attention to the order in which we calculate the derivatives. It is the opposite of the forward pass!
        """
        assert self.cache is not None, "Run a forward pass before the backward pass. Also, don't forget to store the relevat variables\
            such as in 'self.cache = (X, y, ...)"
        dW = None

        ########################################################################
        # TODO:                                                                #
        # Implement the backward pass. Return the gradient w.r.t W --> dW.     #
        # Make sure you've stored ALL needed variables in self.cache. (!!!!)   #
        #                                                                      #
        # Hint 1: It is recommended to follow the TUM article (Section 3) on   #
        # calculating the chain-rule, while dealing with matrix notations:     #
        # https://bit.ly/tum-article                                           #
        #                                                                      #   
        # Hint 2: Remember that the derivative of sigmoid(x) is independent of #
        # x, and could be calculated with the result from the forward pass.    #
        #                                                                      #
        # Hint 3: The argument "dout" stands for the upstream gradeint to this #
        # layer.                                                               #
        ########################################################################

        # 从缓存中取出需要的值
        X, Z = self.cache
        
        # 计算 sigmoid 函数的输出和导数
        sigmoid_output = self.sigmoid(Z)
        sigmoid_grad = sigmoid_output * (1 - sigmoid_output)  # Sigmoid 导数
        
        
        # 链式法则：计算 dL/dZ
        dZ = dout * sigmoid_grad  # 与上游梯度逐元素相乘
        # 确保 dZ 的形状为 (N, 1)，即每个样本一个梯度
        N = X.shape[0]  # 批量大小
        dZ = dZ.reshape(N, 1)  # 确保 dZ 的形状为 (N, 1)
        print('X:' ,dZ.shape)
        # 计算权重矩阵的梯度
        dW = np.dot(X.T, dZ) / dZ.shape[1]  # 对批次大小进行归一化


        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return dW # (dL/dW)

    def sigmoid(self, x):
        """
        Computes the ouput of the sigmoid function.

        :param x: input of the sigmoid, np.array of any shape
        :return: output of the sigmoid with same shape as input vector x
        """
        out = None

        ########################################################################
        # TODO:                                                                #
        # Implement the sigmoid function over the input x. Return "out".       #
        # Note: The sigmoid() function operates element-wise.                  #
        ########################################################################

        # Implement the sigmoid function element-wise
        out = 1 / (1 + np.exp(-x))

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

        return out

    def save_model(self):
        directory = 'models'
        model = {self.model_name: self}
        if not os.path.exists(directory):
            os.makedirs(directory)
        pickle.dump(
            model,
            open(
                directory +
                '/' +
                self.model_name +
                '.p',
                'wb'))
