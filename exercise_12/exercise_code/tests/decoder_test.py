from .base_tests import UnitTest, string_utils, test_results_to_score, CompositeTest
import torch
import numpy as np
from ..util.notebook_util import count_parameters
from ..util.transformer_util import create_causal_mask
from ..tests import tensor_path
from os.path import join


class DecoderBlockOutputShapeTest(UnitTest):
    def __init__(self):
        super().__init__()

        from ..network import DecoderBlock


        batch_size = np.random.randint(low=30, high=100)
        sequence_length = np.random.randint(low=30, high=100)
        d_model = np.random.randint(low=30, high=100)
        d_k = np.random.randint(low=30, high=100)
        d_v = np.random.randint(low=30, high=100)
        n_heads = np.random.randint(low=30, high=100)
        d_ff = np.random.randint(low=30, high=100)
        random_input = torch.rand((batch_size, sequence_length, d_model))
        random_context = torch.rand((batch_size, sequence_length, d_model))
        causal_mask = create_causal_mask(sequence_length)

        decoder_block = DecoderBlock(d_model=d_model,
                                     d_k=d_k,
                                     d_v=d_v,
                                     n_heads=n_heads,
                                     d_ff=d_ff)

        output = decoder_block(random_input, random_context, causal_mask)
        self.result = output.shape
        self.expected = torch.Size([batch_size, sequence_length, d_model])

    def test(self):
        return self.expected == self.result

    def define_failure_message(self):
        return " ".join(f"{self.test_name} {self.failed_msg} {string_utils.ARROW}\
            Expected shape {self.expected}, got shape {self.result}.".split())


class DecoderBlockOutputNorm(UnitTest):
    def __init__(self):
        super().__init__()

        from ..network import DecoderBlock


        batch_size = np.random.randint(low=30, high=100)
        sequence_length = np.random.randint(low=30, high=100)
        d_model = np.random.randint(low=30, high=100)
        d_k = np.random.randint(low=30, high=100)
        d_v = np.random.randint(low=30, high=100)
        n_heads = np.random.randint(low=1, high=10)
        d_ff = np.random.randint(low=30, high=1000)
        random_input = torch.rand((batch_size, sequence_length, d_model))
        random_context = torch.rand((batch_size, sequence_length, d_model))
        causal_mask = create_causal_mask(sequence_length)

        decoder_block = DecoderBlock(d_model=d_model,
                                     d_k=d_k,
                                     d_v=d_v,
                                     n_heads=n_heads,
                                     d_ff=d_ff)

        output = decoder_block(random_input, random_context, causal_mask)
        mean = torch.mean(output).item()
        std = torch.std(output).item()
        self.result = np.array([mean, std])
        self.expected = np.array([0, 1])

    def test(self):
        return np.isclose(self.result, self.expected).all()

    def define_failure_message(self):
        return " ".join(f"{self.test_name} {self.failed_msg} {string_utils.ARROW}\
            Expected [Mean, Std]: {self.expected}, got: {self.result}. Please check the layer normalization!".split())


class DecoderBlockParameterCountTest(UnitTest):
    def __init__(self):
        super().__init__()

        from ..network import DecoderBlock

        d_model = np.random.randint(low=30, high=100)
        d_k = np.random.randint(low=30, high=100)
        d_v = np.random.randint(low=30, high=100)
        n_heads = np.random.randint(low=1, high=10)
        d_ff = np.random.randint(low=30, high=1000)

        decoder_block = DecoderBlock(d_model=d_model,
                                     d_k=d_k,
                                     d_v=d_v,
                                     n_heads=n_heads,
                                     d_ff=d_ff)

        count_ln = 2 * d_model
        count_mh = n_heads * (2 * d_model * d_k + d_model * d_v) + d_model * n_heads * d_v
        count_ffn = d_model * d_ff * 2 + d_model + d_ff

        self.expected = 3 * count_ln + 2 * count_mh + count_ffn
        self.result = count_parameters(decoder_block)

    def test(self):
        return self.result == self.expected

    def define_failure_message(self):
        return " ".join(f"{self.test_name} {self.failed_msg} {string_utils.ARROW}\
            Expected {self.expected} learnable parameters, got {self.result}. Please check your model architecture!".split())
    
    
class DecoderBlockValueTest(UnitTest):
    def __init__(self):
        super().__init__()

        from ..network import DecoderBlock

        task_path = join(tensor_path, 'task_7')

        params = torch.load(join(task_path, 'params.pt'), weights_only=True)
        input = torch.load(join(task_path, 'input.pt'), weights_only=True)
        context = torch.load(join(task_path, 'context.pt'), weights_only=True)
        mask = torch.load(join(task_path, 'mask.pt'), weights_only=True)

        decoder_block = DecoderBlock(**params)

        if decoder_block.causal_multi_head.dropout is None:
            decoder_block.load_state_dict(torch.load(join(task_path, 'model_a.pt'), weights_only=True))
        else:
            decoder_block.load_state_dict(torch.load(join(task_path, 'model_b.pt'), weights_only=True))

        self.result = decoder_block(input, context, mask)
        self.expected = torch.load(join(task_path, 'output.pt'), weights_only=True)
    
    def test(self):
        return torch.allclose(self.expected, self.result, rtol=1e-03, atol=1e-05)
    
    def define_failure_message(self):
        return " ".join(f"{self.test_name} {self.failed_msg} {string_utils.ARROW}\
            DecoderBlock output is not correct. ".split())


class DecoderOutputShapeTest(UnitTest):
    def __init__(self):
        super().__init__()

        from ..network import Decoder


        batch_size = np.random.randint(low=30, high=100)
        sequence_length = np.random.randint(low=30, high=100)
        d_model = np.random.randint(low=30, high=100)
        d_k = np.random.randint(low=30, high=100)
        d_v = np.random.randint(low=30, high=100)
        n_heads = np.random.randint(low=1, high=10)
        d_ff = np.random.randint(low=30, high=1000)
        n = np.random.randint(low=1, high=10)
        random_input = torch.rand((batch_size, sequence_length, d_model))
        random_context = torch.rand((batch_size, sequence_length, d_model))

        encoder_stack = Decoder(d_model=d_model,
                                d_k=d_k,
                                d_v=d_v,
                                n_heads=n_heads,
                                d_ff=d_ff,
                                n=n)

        output = encoder_stack(random_input, random_context)
        self.result = output.shape
        self.expected = torch.Size([batch_size, sequence_length, d_model])

    def test(self):
        return self.expected == self.result

    def define_failure_message(self):
        return " ".join(f"{self.test_name} {self.failed_msg} {string_utils.ARROW}\
            Expected shape {self.expected}, got shape {self.result}.".split())


class DecoderParameterCountTest(UnitTest):
    def __init__(self):
        super().__init__()

        from ..network import Decoder

        d_model = np.random.randint(low=30, high=100)
        d_k = np.random.randint(low=30, high=100)
        d_v = np.random.randint(low=30, high=100)
        n_heads = np.random.randint(low=1, high=10)
        d_ff = np.random.randint(low=30, high=1000)
        n = np.random.randint(low=1, high=10)

        decoder_stack = Decoder(d_model=d_model,
                                d_k=d_k,
                                d_v=d_v,
                                n_heads=n_heads,
                                d_ff=d_ff,
                                n=n)

        count_ln = 2 * d_model
        count_mh = n_heads * (2 * d_model * d_k + d_model * d_v) + d_model * n_heads * d_v
        count_ffn = d_model * d_ff * 2 + d_model + d_ff

        self.expected = n * (3 * count_ln + 2 * count_mh + count_ffn)
        self.result = count_parameters(decoder_stack)

    def test(self):
        return self.result == self.expected

    def define_failure_message(self):
        return " ".join(f"{self.test_name} {self.failed_msg} {string_utils.ARROW}\
            Expected {self.expected} learnable parameters, got {self.result}. Please check your model architecture!".split())


class TestTask7(CompositeTest):
    def define_tests(self, ):
        return [
            DecoderBlockOutputShapeTest(),
            DecoderBlockOutputNorm(),
            DecoderBlockParameterCountTest(),
            DecoderBlockValueTest()
        ]


class TestTask8(CompositeTest):
    def define_tests(self, ):
        return [
            DecoderOutputShapeTest(),
            DecoderParameterCountTest()
        ]


def test_task_7():
    test = TestTask7()
    return test_results_to_score(test())


def test_task_8():
    test = TestTask8()
    return test_results_to_score(test())
