import torch

from .base_tests import UnitTest, string_utils, test_results_to_score, CompositeTest
import numpy as np
import torch
from ..util.notebook_util import count_parameters


class TestLinear1(UnitTest):
    def __init__(self):
        super().__init__()

        from ..network import FeedForwardNeuralNetwork


        d_model = np.random.randint(low=30, high=100)
        d_ff = np.random.randint(low=30, high=100)

        ffn = FeedForwardNeuralNetwork(d_model=d_model, d_ff=d_ff)

        self.result = ffn.linear_1.weight.T.shape
        self.expected = torch.Size([d_model, d_ff])

    def test(self):
        return self.expected == self.result

    def define_failure_message(self):
        return " ".join(f"{self.test_name} {self.failed_msg} {string_utils.ARROW}\
            Expected shape {self.expected}, got shape {self.result}.".split())


class TestLinear1Bias(UnitTest):
    def __init__(self):
        super().__init__()

        from ..network import FeedForwardNeuralNetwork


        d_model = np.random.randint(low=30, high=100)
        d_ff = np.random.randint(low=30, high=100)

        ffn = FeedForwardNeuralNetwork(d_model=d_model, d_ff=d_ff)
        self.result = ffn.linear_1.bias.shape
        self.expected = torch.Size([d_ff])

    def test(self):
        return self.expected == self.result

    def define_failure_message(self):
        return " ".join(f"{self.test_name} {self.failed_msg} {string_utils.ARROW}\
            Expected shape {self.expected}, got shape {self.result}.".split())


class TestLinear2(UnitTest):
    def __init__(self):
        super().__init__()

        from ..network import FeedForwardNeuralNetwork


        d_model = np.random.randint(low=30, high=100)
        d_ff = np.random.randint(low=30, high=100)

        ffn = FeedForwardNeuralNetwork(d_model=d_model, d_ff=d_ff)
        self.result = ffn.linear_2.weight.T.shape
        self.expected = torch.Size([d_ff, d_model])

    def test(self):
        return self.expected == self.result

    def define_failure_message(self):
        return " ".join(f"{self.test_name} {self.failed_msg} {string_utils.ARROW}\
            Expected shape {self.expected}, got shape {self.result}.".split())


class TestLinear2Bias(UnitTest):
    def __init__(self):
        super().__init__()

        from ..network import FeedForwardNeuralNetwork


        d_model = np.random.randint(low=30, high=100)
        d_ff = np.random.randint(low=30, high=100)

        ffn = FeedForwardNeuralNetwork(d_model=d_model, d_ff=d_ff)
        self.result = ffn.linear_2.bias.shape
        self.expected = torch.Size([d_model])

    def test(self):
        return self.expected == self.result

    def define_failure_message(self):
        return " ".join(f"{self.test_name} {self.failed_msg} {string_utils.ARROW}\
            Expected shape {self.expected}, got shape {self.result}.".split())


class TestParameterCount(UnitTest):
    def __init__(self):
        super().__init__()

        from ..network import FeedForwardNeuralNetwork

        
        d_model = np.random.randint(low=30, high=100)
        d_ff = np.random.randint(low=30, high=100)

        ffn = FeedForwardNeuralNetwork(d_model=d_model, d_ff=d_ff)
        self.expected = d_model * d_ff * 2 + d_model + d_ff
        self.result = count_parameters(ffn)

    def test(self):
        return self.result == self.expected

    def define_failure_message(self):
        return " ".join(f"{self.test_name} {self.failed_msg} {string_utils.ARROW}\
            Expected {self.expected} learnable parameters, got {self.result}. Please check your model architecture!".split())


class TestTask3(CompositeTest):
    def define_tests(self):
        return [
            TestLinear1(),
            TestLinear1Bias(),
            TestLinear2(),
            TestLinear2Bias(),
            TestParameterCount()
        ]


def test_task_3():
    test = TestTask3()
    return test_results_to_score(test())
