from .base_tests import UnitTest, string_utils, test_results_to_score, CompositeTest
import torch
import numpy as np
from ..tests import tensor_path
from os.path import join
import pickle

class EmbeddingShapeTest(UnitTest):
    def __init__(self):
        super().__init__()

        from ..network import Embedding

        vocab_size = np.random.randint(low=30, high=100)
        d_model = np.random.randint(low=30, high=50) * 2
        max_length = np.random.randint(low=30, high=100)
        embedding = Embedding(vocab_size=vocab_size, d_model=d_model, max_length=max_length)
        self.result = embedding.embedding.weight.shape
        self.expected = torch.Size([vocab_size, d_model])

    def test(self):
        return self.expected == self.result

    def define_failure_message(self):
        return " ".join(f"{self.test_name} {self.failed_msg} {string_utils.ARROW}\
            Expected shape {self.expected}, got shape {self.result}.".split())


class EmbeddingForwardShapeTest(UnitTest):
    def __init__(self):
        super().__init__()

        from ..network import Embedding

        vocab_size = np.random.randint(low=30, high=100)
        d_model = np.random.randint(low=30, high=50) * 2
        max_length = np.random.randint(low=30, high=100)
        batch_size = np.random.randint(low=30, high=100)
        sequence_length = np.random.randint(low=1, high=max_length)
        embedding = Embedding(vocab_size, d_model, max_length)

        self.result = embedding(torch.ones((batch_size, sequence_length), dtype=torch.int)).shape
        self.expected = torch.Size([batch_size, sequence_length, d_model])

    def test(self):
        return self.expected == self.result

    def define_failure_message(self):
        return " ".join(f"{self.test_name} {self.failed_msg} {string_utils.ARROW}\
            Expected shape {self.expected}, got shape {self.result}.".split())


class PositionalEncodingValueTest(UnitTest):
    def __init__(self):
        super().__init__()
        
        from ..network.embedding import positional_encoding
        task_path = join(tensor_path, 'task_4')

        params = torch.load(join(task_path, 'params.pt'), weights_only=True)

        self.result = positional_encoding(**params)
        self.expected = torch.load(join(task_path, 'output.pt'), weights_only=True)

    def test(self):
        return torch.allclose(self.expected, self.result, atol=1e-5)

    def define_failure_message(self):
        return " ".join(f"{self.test_name} {self.failed_msg} {string_utils.ARROW}\
            Positional Encoding wasn't implemented correctly!.".split())
    
class EmbeddingValueTest(UnitTest):
    def __init__(self):
        super().__init__()

        from ..network import Embedding
        task_path = join(tensor_path, 'task_1')
        input = torch.load(join(task_path, 'input.pt'), weights_only=True)

        params = torch.load(join(task_path, 'params.pt'), weights_only=True)
        embedding = Embedding(vocab_size=params['vocab_size'], 
                              d_model=params['d_model'], 
                              max_length=params['max_length'])
        
        embedding.embedding.weight = torch.load(join(task_path, 'embedding.pt'), weights_only=True)
        
        if embedding.pos_encoding is None:
            self.expected = torch.load(join(task_path, 'output_a.pt'), weights_only=True)
        else:
            self.expected = torch.load(join(task_path, 'output_b.pt'), weights_only=True)

        self.result = embedding(input)

    def test(self):
        return torch.allclose(self.expected, self.result, atol=1e-5)

    def define_failure_message(self):
        return " ".join(f"{self.test_name} {self.failed_msg} {string_utils.ARROW}\
            Embedding Layer wasn't implemented correctly!.".split())

                

class TestTask1(CompositeTest):
    def define_tests(self, ):
        return [
            EmbeddingShapeTest(),
            EmbeddingForwardShapeTest(),
            EmbeddingValueTest()
        ]


class TestTask4(CompositeTest):
    def define_tests(self, ):
        return [
            PositionalEncodingValueTest(),
            EmbeddingForwardShapeTest(),
            EmbeddingValueTest()
        ]


def test_task_1():
    test = TestTask1()
    return test_results_to_score(test())


def test_task_4():
    test = TestTask4()
    return test_results_to_score(test())
