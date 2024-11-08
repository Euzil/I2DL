
import os
import pickle
import numpy as np
# from exercise_code.networks.linear_model import *


class Loss(object):
    def __init__(self):
        self.grad_history = []

    def forward(self, y_out, y_truth, individual_losses=False):
        return NotImplementedError

    def backward(self, y_out, y_truth, upstream_grad=1.):
        return NotImplementedError

    def __call__(self, y_out, y_truth, individual_losses=False):
        loss = self.forward(y_out, y_truth, individual_losses)
        return loss


class L1(Loss):

    def forward(self, y_out, y_truth, individual_losses=False):
        """
        Performs the forward pass of the L1 loss function.

        :param y_out: [N, ] array predicted value of your model.
        :y_truth: [N, ] array ground truth value of your training set.
        :return: 
            - individual_losses=False --> A single scalar, which is the mean of the binary cross entropy loss 
                for each sample of your training set.
            - individual_losses=True  --> [N, ] array of binary cross entropy loss for each sample of your training set.
        """
        result = None

        result = np.abs(y_out - y_truth)
        
        if individual_losses:
            return result
        return np.mean(result)

    def backward(self, y_out, y_truth):
        """
        Performs the backward pass of the L1 loss function.

        :param y_out: [N, ] array predicted value of your model.
        :y_truth: [N, ] array ground truth value of your training set.
        :return: [N, ] array of L1 loss gradients w.r.t y_out for each sample of your training set.
        """
        gradient = None
        gradient = y_out - y_truth

        zero_loc = np.where(gradient == 0)
        negative_loc = np.where(gradient < 0)
        positive_loc = np.where(gradient > 0)

        gradient[zero_loc] = 0
        gradient[positive_loc] = 1
        gradient[negative_loc] = -1
    
        return gradient / len(y_out)


class MSE(Loss):

    def forward(self, y_out, y_truth, individual_losses=False):
        """
        Performs the forward pass of the MSE loss function.

        :param y_out: [N, ] array predicted value of your model.
        :y_truth: [N, ] array ground truth value of your training set.
        :return: 
            - individual_losses=False --> A single scalar, which is the mean of the binary cross entropy loss 
                for each sample of your training set.
            - individual_losses=True  --> [N, ] array of binary cross entropy loss for each sample of your training set.
        """
        result = None
        
        result = (y_out - y_truth) ** 2
        
        if individual_losses:
            return result
        return np.mean(result)

    def backward(self, y_out, y_truth):
        """
        Performs the backward pass of the MSE loss function.

        :param y_out: [N, ] array predicted value of your model.
        :y_truth: [N, ] array ground truth value of your training set.
        :return: [N, ] array of MSE loss gradients w.r.t y_out for each sample of your training set.
        """
        gradient = None
        gradient = 2 * (y_out - y_truth) / len(y_out)
        return gradient


class BCE(Loss):

    def forward(self, y_out, y_truth, individual_losses=False):
        """
        Performs the forward pass of the binary cross entropy loss function.

        :param y_out: [N, ] array predicted value of your model (the Logits).
        :y_truth: [N, ] array ground truth value of your training set.
        :return: 
            - individual_losses=False --> A single scalar, which is the mean of the binary cross entropy loss 
                for each sample of your training set.
            - individual_losses=True  --> [N, ] array of binary cross entropy loss values for each sample of your batch.
        """
        result = None
        ########################################################################
        # TODO:                                                                #
        # Implement the forward pass and return the output of the BCE loss     #
        # for each imstance in the batch.                                      #
        #                                                                      #
        ########################################################################

        # 对 logits 应用 sigmoid 函数，得到预测的概率
        y_pred = 1 / (1 + np.exp(-y_out))  # 将y_out应用sigmoid


        # 计算每个样本的二元交叉熵损失
        result = -(y_truth * np.log(y_out) + (1 - y_truth) * np.log(1 - y_out))

        # 如果需要返回每个样本的损失，直接返回
        if individual_losses:
            print(result,'result')
            return result

        # 返回所有样本的平均损失
        return np.mean(result)

    def backward(self, y_out, y_truth):
        """
        Performs the backward pass of the loss function.

        :param y_out: [N, ] array predicted value of your model.
        :y_truth: [N, ] array ground truth value of your training set.
        :return: [N, ] array of binary cross entropy loss gradients w.r.t y_out 
                for each sample of your training set.
        """
        gradient = None

        ########################################################################
        # TODO:                                                                #
        # Implement the backward pass. Return the gradient w.r.t to the input  #
        # to the loss function, y_out.                                         #
        #                                                                      #
        # Hint:                                                                #
        #   Don't forget to divide by N, which is the number of samples in     #
        #   the batch. It is crucial for the magnitude of the gradient.        #
        ########################################################################

        gradient = -(y_truth / y_out - (1 - y_truth) / (1 - y_out))

        # 通过N进行归一化
        N = y_out.shape[0]
        gradient /= N

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return gradient


