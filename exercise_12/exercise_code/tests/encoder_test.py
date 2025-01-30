from .base_tests import UnitTest, string_utils, test_results_to_score, CompositeTest
import torch
import numpy as np
from ..util.notebook_util import count_parameters
from ..tests import tensor_path
from os.path import join


class EncoderBlockOutputShapeTest(UnitTest):
    def __init__(self):
        super().__init__()

        from ..network import EncoderBlock

        batch_size = np.random.randint(low=30, high=100)
        sequence_length = np.random.randint(low=30, high=100)
        d_model = np.random.randint(low=30, high=100)
        d_k = np.random.randint(low=30, high=100)
        d_v = np.random.randint(low=30, high=100)
        n_heads = np.random.randint(low=30, high=100)
        d_ff = np.random.randint(low=30, high=100)
        random_input = torch.rand((batch_size, sequence_length, d_model))

        encoder_block = EncoderBlock(d_model=d_model,
                                          d_k=d_k,
                                          d_v=d_v,
                                          n_heads=n_heads,
                                          d_ff=d_ff)

        output = encoder_block(random_input, pad_mask=None)

        self.result = output.shape
        self.expected = torch.Size([batch_size, sequence_length, d_model])

    def test(self):
        return self.expected == self.result

    def define_failure_message(self):
        return " ".join(f"{self.test_name} {self.failed_msg} {string_utils.ARROW}\
            Expected shape {self.expected}, got shape {self.result}.".split())


class EncoderBlockOutputNorm(UnitTest):
    def __init__(self):
        super().__init__()

        from ..network import EncoderBlock


        batch_size = np.random.randint(low=30, high=100)
        sequence_length = np.random.randint(low=30, high=100)
        d_model = np.random.randint(low=30, high=100)
        d_k = np.random.randint(low=30, high=100)
        d_v = np.random.randint(low=30, high=100)
        n_heads = np.random.randint(low=1, high=10)
        d_ff = np.random.randint(low=30, high=1000)
        dropout = 0
        random_input = torch.rand((batch_size, sequence_length, d_model))

        encoder_block = EncoderBlock(d_model=d_model,
                                          d_k=d_k,
                                          d_v=d_v,
                                          n_heads=n_heads,
                                          d_ff=d_ff)

        output = encoder_block(random_input)
        mean = torch.mean(output).item()
        std = torch.std(output).item()
        self.result = np.array([mean, std])
        self.expected = np.array([0, 1])

    def test(self):
        return np.isclose(self.result, self.expected).all()

    def define_failure_message(self):
        return " ".join(f"{self.test_name} {self.failed_msg} {string_utils.ARROW}\
            Expected [Mean, Std]: {self.expected}, got: {self.result}. Please check the layer normalization!".split())


class EncoderBlockParameterCountTest(UnitTest):
    def __init__(self):
        super().__init__()

        from ..network import EncoderBlock

        d_model = np.random.randint(low=30, high=100)
        d_k = np.random.randint(low=30, high=100)
        d_v = np.random.randint(low=30, high=100)
        n_heads = np.random.randint(low=1, high=10)
        d_ff = np.random.randint(low=30, high=1000)

        encoder_block = EncoderBlock(d_model=d_model,
                                          d_k=d_k,
                                          d_v=d_v,
                                          n_heads=n_heads,
                                          d_ff=d_ff)

        count_ln = 2 * d_model
        count_mh = n_heads * (
                2 * d_model * d_k + d_model * d_v) + d_model * n_heads * d_v
        count_ffn = d_model * d_ff * 2 + d_model + d_ff

        self.expected = 2 * count_ln + count_mh + count_ffn
        self.result = count_parameters(encoder_block)

    def test(self):
        return self.result == self.expected

    def define_failure_message(self):
        return " ".join(f"{self.test_name} {self.failed_msg} {string_utils.ARROW}\
            Expected {self.expected} learnable parameters, got {self.result}. Please check your model architecture!".split())
    
class EncoderBlockValueTest(UnitTest):
    def __init__(self):
        super().__init__()

        from ..network import EncoderBlock

        task_path = join(tensor_path, 'task_4')

        params = torch.load(join(task_path, 'params.pt'), weights_only=True)
        input = torch.load(join(task_path, 'input.pt'), weights_only=True)
        encoder_block = EncoderBlock(**params)

        if encoder_block.multi_head.dropout is None:
            encoder_block.load_state_dict(torch.load(join(task_path, 'model_a.pt'), weights_only=True))
        else:
            encoder_block.load_state_dict(torch.load(join(task_path, 'model_b.pt'), weights_only=True))

        self.result = encoder_block(input)
        self.expected = torch.load(join(task_path, 'output.pt'), weights_only=True)
    
    def test(self):
        return torch.allclose(self.expected, self.result, rtol=1e-03, atol=1e-05)
    
    def define_failure_message(self):
        return " ".join(f"{self.test_name} {self.failed_msg} {string_utils.ARROW}\
            EncoderBlock output is not correct. ".split())


class EncoderOutputShapeTest(UnitTest):
    def __init__(self):
        super().__init__()

        from ..network import Encoder


        batch_size = np.random.randint(low=30, high=100)
        sequence_length = np.random.randint(low=30, high=100)
        d_model = np.random.randint(low=30, high=100)
        d_k = np.random.randint(low=30, high=100)
        d_v = np.random.randint(low=30, high=100)
        n_heads = np.random.randint(low=1, high=10)
        d_ff = np.random.randint(low=30, high=1000)
        n = np.random.randint(low=1, high=10)
        random_input = torch.rand((batch_size, sequence_length, d_model))

        encoder_stack = Encoder(d_model=d_model,
                                          d_k=d_k,
                                          d_v=d_v,
                                          n_heads=n_heads,
                                          d_ff=d_ff,
                                          n=n)

        output = encoder_stack(random_input)
        self.result = output.shape
        self.expected = torch.Size([batch_size, sequence_length, d_model])

    def test(self):
        return self.expected == self.result

    def define_failure_message(self):
        return " ".join(f"{self.test_name} {self.failed_msg} {string_utils.ARROW}\
            Expected shape {self.expected}, got shape {self.result}.".split())


class EncoderParameterCountTest(UnitTest):
    def __init__(self):
        super().__init__()

        from ..network import Encoder

        d_model = np.random.randint(low=30, high=100)
        d_k = np.random.randint(low=30, high=100)
        d_v = np.random.randint(low=30, high=100)
        n_heads = np.random.randint(low=1, high=10)
        d_ff = np.random.randint(low=30, high=1000)
        n = np.random.randint(low=1, high=10)

        encoder_stack = Encoder(d_model=d_model,
                                          d_k=d_k,
                                          d_v=d_v,
                                          n_heads=n_heads,
                                          d_ff=d_ff,
                                          n=n)

        count_ln = 2 * d_model
        count_mh = n_heads * (
                2 * d_model * d_k + d_model * d_v) + d_model * n_heads * d_v
        count_ffn = d_model * d_ff * 2 + d_model + d_ff

        self.expected = n * (2 * count_ln + count_mh + count_ffn)
        self.result = count_parameters(encoder_stack)

    def test(self):
        return self.result == self.expected

    def define_failure_message(self):
        return " ".join(f"{self.test_name} {self.failed_msg} {string_utils.ARROW}\
            Expected {self.expected} learnable parameters, got {self.result}. Please check your model architecture!".split())


class TestTask4(CompositeTest):
    def define_tests(self, ):
        return [
            EncoderBlockOutputShapeTest(),
            EncoderBlockOutputNorm(),
            EncoderBlockParameterCountTest(),
            EncoderBlockValueTest()
        ]


class TestTask5(CompositeTest):
    def define_tests(self, ):
        return [
            EncoderOutputShapeTest(),
            EncoderParameterCountTest()
        ]


def test_task_4():
    test = TestTask4()
    return test_results_to_score(test())


def test_task_5():
    test = TestTask5()
    return test_results_to_score(test())
