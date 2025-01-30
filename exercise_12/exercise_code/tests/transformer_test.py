from .base_tests import UnitTest, string_utils, test_results_to_score, CompositeTest
import torch
import numpy as np
from ..util.notebook_util import count_parameters
import os
import copy
from ..tests import tensor_path
from os.path import join


class TransformerOutputShapeTest(UnitTest):
    def __init__(self):
        super().__init__()

        from ..network import Transformer

        vocab_size = np.random.randint(low=30, high=100)
        batch_size = np.random.randint(low=30, high=100)
        sequence_length = np.random.randint(low=30, high=100)
        d_model = np.random.randint(low=30, high=50) * 2
        d_k = np.random.randint(low=30, high=100)
        d_v = np.random.randint(low=30, high=100)
        n_heads = np.random.randint(low=1, high=10)
        d_ff = np.random.randint(low=30, high=1000)
        n = np.random.randint(low=1, high=10)
        dropout = 0
        random_input = torch.randint(low=0, high=vocab_size, size=(batch_size, sequence_length))
        random_context = torch.randint(low=0, high=vocab_size, size=(batch_size, sequence_length))

        hparams = {
            'd_model': d_model,
            'd_k': d_k,
            'd_v': d_v,
            'n_heads': n_heads,
            'd_ff': d_ff,
            'n': n,
            'dropout': dropout
        }

        transformer = Transformer(vocab_size=vocab_size,
                                  eos_token_id=0,
                                  hparams=hparams)

        output = transformer(random_input, random_context)

        self.result = output.shape
        self.expected = torch.Size([batch_size, sequence_length, vocab_size])

    def test(self):
        return self.expected == self.result

    def define_failure_message(self):
        return " ".join(f"{self.test_name} {self.failed_msg} {string_utils.ARROW}\
            Expected shape {self.expected}, got shape {self.result}.".split())


class TransformerParameterCountWeightTyingTest(UnitTest):
    def __init__(self):
        super().__init__()

        from ..network import Transformer

        vocab_size = np.random.randint(low=30, high=100)
        d_model = np.random.randint(low=30, high=50) * 2
        d_k = np.random.randint(low=30, high=100)
        d_v = np.random.randint(low=30, high=100)
        n_heads = np.random.randint(low=1, high=10)
        d_ff = np.random.randint(low=30, high=1000)
        n = np.random.randint(low=1, high=10)
        dropout = 0

        hparams = {
            'd_model': d_model,
            'd_k': d_k,
            'd_v': d_v,
            'n_heads': n_heads,
            'd_ff': d_ff,
            'n': n,
            'dropout': dropout
        }

        transformer = Transformer(vocab_size=vocab_size,
                                  eos_token_id=0,
                                  hparams=hparams)

        count_ln = 2 * d_model
        count_mh = n_heads * (2 * d_model * d_k + d_model * d_v) + d_model * n_heads * d_v
        count_ffn = d_model * d_ff * 2 + d_model + d_ff

        self.expected = n * (5 * count_ln + 3 * count_mh + 2 * count_ffn) + vocab_size * d_model
        self.result = count_parameters(transformer)

    def test(self):
        return self.result == self.expected

    def define_failure_message(self):
        return " ".join(f"{self.test_name} {self.failed_msg} {string_utils.ARROW}\
            Expected {self.expected} learnable parameters, got {self.result}. Please check your model architecture!".split())


class TransformerParameterCountTest(UnitTest):
    def __init__(self):
        super().__init__()

        from ..network import Transformer

        vocab_size = np.random.randint(low=30, high=100)
        d_model = np.random.randint(low=30, high=50) * 2
        d_k = np.random.randint(low=30, high=100)
        d_v = np.random.randint(low=30, high=100)
        n_heads = np.random.randint(low=1, high=10)
        d_ff = np.random.randint(low=30, high=1000)
        n = np.random.randint(low=1, high=10)
        dropout = 0

        hparams = {
            'd_model': d_model,
            'd_k': d_k,
            'd_v': d_v,
            'n_heads': n_heads,
            'd_ff': d_ff,
            'n': n,
            'dropout': dropout
        }

        transformer = Transformer(vocab_size=vocab_size,
                                  eos_token_id=0,
                                  hparams=hparams,
                                  weight_tying=False)

        count_ln = 2 * d_model
        count_mh = n_heads * (2 * d_model * d_k + d_model * d_v) + d_model * n_heads * d_v
        count_ffn = d_model * d_ff * 2 + d_model + d_ff

        self.expected = n * (5 * count_ln + 3 * count_mh + 2 * count_ffn) + 2 * vocab_size * d_model
        self.result = count_parameters(transformer)

    def test(self):
        return self.result == self.expected

    def define_failure_message(self):
        return " ".join(f"{self.test_name} {self.failed_msg} {string_utils.ARROW}\
            Expected {self.expected} learnable parameters, got {self.result}. Please check your model architecture!".split())
    
class TransformerValueTest(UnitTest):
    def __init__(self):
        super().__init__()

        from ..network import Transformer
        task_path = join(tensor_path, 'task_9')

        params = torch.load(join(task_path, 'params.pt'), weights_only=True)
        input = torch.load(join(task_path, 'input.pt'), weights_only=True)
        context = torch.load(join(task_path, 'context.pt'), weights_only=True)

        transformer = Transformer(**params)

        if transformer.embedding.dropout is None:
            transformer.load_state_dict(torch.load(join(task_path, 'model_a.pt'), weights_only=True))
        else:
            transformer.load_state_dict(torch.load(join(task_path, 'model_b.pt'), weights_only=True))
        
        self.result = transformer(input, context)
        self.expected = torch.load(join(task_path, 'output.pt'), weights_only=True)

    def test(self):
        return torch.allclose(self.expected, self.result, rtol=1e-03, atol=1e-05)
    
    def define_failure_message(self):
        return " ".join(f"{self.test_name} {self.failed_msg} {string_utils.ARROW}\
            The output of your Transformer is not correct. Please check your implementation. Hint: Check the order of inputs!".split())


class TransformerPaddingTest(UnitTest):
    def __init__(self):
        super().__init__()

        from ..network import Transformer

        vocab_size = np.random.randint(low=10, high=50)
        batch_size = np.random.randint(low=5, high=20)

        sequence_length_encoder = np.random.randint(low=10, high=50)
        sequence_length_decoder = np.random.randint(low=10, high=50)

        sequence_lengths_encoder = np.random.randint(low=5, high=sequence_length_encoder,
                                                     size=(batch_size,))
        sequence_lengths_decoder = np.random.randint(low=5, high=sequence_length_decoder,
                                                     size=(batch_size,))

        d_model = np.random.randint(low=10, high=25) * 2
        d_k = np.random.randint(low=10, high=50)
        d_v = np.random.randint(low=10, high=50)
        n_heads = np.random.randint(low=1, high=5)
        d_ff = np.random.randint(low=10, high=50)
        n = np.random.randint(low=1, high=5)
        dropout = 0

        random_input_encoder = torch.randint(low=0, high=vocab_size,
                                             size=(batch_size, sequence_length_encoder))
        random_input_decoder = torch.randint(low=0, high=vocab_size,
                                             size=(batch_size, sequence_length_decoder))

        random_mask_encoder = torch.tensor([[i < length for i in range(sequence_length_encoder)] for length in
                                            sequence_lengths_encoder])
        random_mask_decoder = torch.tensor([[i < length for i in range(sequence_length_decoder)] for length in
                                            sequence_lengths_decoder])

        hparams = {
            'd_model': d_model,
            'd_k': d_k,
            'd_v': d_v,
            'n_heads': n_heads,
            'd_ff': d_ff,
            'n': n,
            'dropout': dropout
        }

        transformer = Transformer(vocab_size=vocab_size,
                                  eos_token_id=0,
                                  hparams=hparams)

        output_batched = transformer(random_input_encoder,
                                     random_input_decoder,
                                     random_mask_encoder.unsqueeze(-2).bool(),
                                     random_mask_decoder.unsqueeze(-2).bool())

        outputs_single = []
        outputs_batched = []
        for i in range(batch_size):
            encoder_input = (random_input_encoder[i][random_mask_encoder[i]]).unsqueeze(0)
            decoder_input = (random_input_decoder[i][random_mask_decoder[i]]).unsqueeze(0)
            outputs_single.append(transformer(encoder_input, decoder_input))
            outputs_batched.append(output_batched[i][random_mask_decoder[i]])

        self.result = all([torch.allclose(batch, single, atol=1e-2) for batch, single in
                           zip(outputs_batched, outputs_single)])

    def test(self):
        return self.result

    def define_failure_message(self):
        return " ".join(f"{self.test_name} {self.failed_msg} {string_utils.ARROW}\
            There seem to be some mistakes in your padding implementation!".split())


class AttentionDropoutTest(UnitTest):
    def __init__(self):
        super().__init__()

        from ..network import ScaledDotAttention, SCORE_SAVER

        batch_size = 50
        d_model = 50

        dropout = 0.5
        random_input = torch.rand(size=(batch_size, d_model, d_model))

        attention_head_no_dropout = ScaledDotAttention(d_k=d_model, dropout=0)

        attention_head_dropout = ScaledDotAttention(d_k=d_model, dropout=dropout)

        SCORE_SAVER.record_scores()
        attention_head_dropout(random_input, random_input, random_input)
        attention_head_no_dropout(random_input, random_input, random_input)

        scores = SCORE_SAVER.get_scores()
        scores_dropout = scores[0] * (1 - dropout)
        scores_no_dropout = scores[1]

        self.result = torch.abs(scores_dropout - scores_no_dropout)
        self.result = len(self.result[self.result > 1e-4]) / (batch_size * d_model * d_model)
        self.expected = dropout

    def test(self):
        return np.isclose(self.result, self.expected, atol=1e-1)

    def define_failure_message(self):
        return " ".join(f"{self.test_name} {self.failed_msg} {string_utils.ARROW}\
            There seem to be some mistakes in your dropout implementation in ScaledDotAttention!".split())


class EmbeddingDropoutTest(UnitTest):
    def __init__(self):
        super().__init__()

        from ..network import Embedding

        batch_size = 50
        max_length = 100
        sequence_length = 100
        vocab_size = 100
        d_model = 100

        dropout = 0.5
        embedding = Embedding(vocab_size=vocab_size, d_model=d_model, max_length=max_length, dropout=dropout)

        random_input = torch.randint(0, vocab_size, size=(batch_size, sequence_length))
        outputs = embedding(random_input)
        self.result = torch.abs(outputs)
        self.result = len(self.result[self.result < 1e-4]) / (batch_size * sequence_length * d_model)
        self.expected = dropout

    def test(self):
        return np.isclose(self.result, self.expected, atol=1e-1)

    def define_failure_message(self):
        return " ".join(f"{self.test_name} {self.failed_msg} {string_utils.ARROW}\
            There seem to be some mistakes in your dropout implementation in Embedding!".split())


class MultiHeadDropoutTest(UnitTest):
    def __init__(self):
        super().__init__()

        from ..network import MultiHeadAttention
        
        batch_size = 50
        sequence_length = 100
        d_model = 100
        d_k = 100
        d_v = 100
        n_heads = 10
        dropout = 0.5
        multi_head_attention = MultiHeadAttention(d_model=d_model,
                                                  d_k=d_k,
                                                  d_v=d_v,
                                                  n_heads=n_heads,
                                                  dropout=dropout)

        random_input = torch.rand(size=(batch_size, sequence_length, d_model))
        outputs = multi_head_attention(random_input, random_input, random_input)
        self.result = torch.abs(outputs)
        self.result = len(self.result[self.result < 1e-4]) / (batch_size * sequence_length * d_model)
        self.expected = dropout

    def test(self):
        return np.isclose(self.result, self.expected, atol=1e-1)

    def define_failure_message(self):
        return " ".join(f"{self.test_name} {self.failed_msg} {string_utils.ARROW}\
            There seem to be some mistakes in your dropout implementation in MultiHeadAttention!".split())


class FeedForwardNeuralNetworkDropoutTest(UnitTest):
    def __init__(self):
        super().__init__()

        from ..network import FeedForwardNeuralNetwork
        
        batch_size = 50
        sequence_length = 100
        d_model = 100
        d_ff = 100
        dropout = 0.5
        ffn = FeedForwardNeuralNetwork(d_model=d_model,
                                       d_ff=d_ff,
                                       dropout=dropout)

        random_input = torch.rand(size=(batch_size, sequence_length, d_model))
        outputs = ffn(random_input)
        self.result = torch.abs(outputs)
        self.result = len(self.result[self.result < 1e-4]) / (batch_size * sequence_length * d_model)
        self.expected = dropout

    def test(self):
        return np.isclose(self.result, self.expected, atol=1e-1)

    def define_failure_message(self):
        return " ".join(f"{self.test_name} {self.failed_msg} {string_utils.ARROW}\
            There seem to be some mistakes in your dropout implementation in FeedForwardNeuralNetwork!".split())


class TestTask9(CompositeTest):
    def define_tests(self, ):
        return [
            TransformerOutputShapeTest(),
            TransformerParameterCountTest(),
            TransformerParameterCountWeightTyingTest(),
            TransformerValueTest()
        ]


class TestTask10(CompositeTest):
    def define_tests(self, ):
        return [
            TransformerPaddingTest()
        ]


class TestTask11(CompositeTest):
    def define_tests(self, ):
        return [EmbeddingDropoutTest(),
                AttentionDropoutTest(),
                MultiHeadDropoutTest(),
                FeedForwardNeuralNetworkDropoutTest()]


class TestModelParameters(UnitTest):
    def __init__(self, model):
        super().__init__()
        model = model
        self.result = count_parameters(model)
        self.expected = 5000000

    def test(self):
        if self.result < self.expected:
            print(f"Your model has {self.result} parameters.")
            return True
        return False

    def define_failure_message(self):
        return " ".join(f"{self.test_name} {self.failed_msg} {string_utils.ARROW}\
            Your model has {self.result} parameters. Make sure you stay under {self.expected}".split())


class TestModelAccuracy(UnitTest):
    def __init__(self, trainer):
        super().__init__()
        trainer = trainer
        trainer._eval_loop()
        self.result = trainer.val_metrics.get_epoch_acc()
        self.expected = 0.5

    def test(self):
        return self.result > self.expected

    def define_failure_message(self):
        return " ".join(f"{self.test_name} {self.failed_msg} {string_utils.ARROW}\
            Your model has an accuracy of only {self.result * 100:.2f}%. To finish this task, you need to reach at least {self.expected * 100}%".split())


def test_task_9():
    test = TestTask9()
    return test_results_to_score(test())


def test_task_10():
    test = TestTask10()
    return test_results_to_score(test())


def test_task_11():
    test = TestTask11()
    return test_results_to_score(test())


def test_model_parameters(model):
    test = TestModelParameters(model)
    return test_results_to_score(test())


def test_and_save_model(trainer, tokenizer, model_path):
    trainer_copy = copy.deepcopy(trainer)
    model_copy = trainer_copy.model
    device = trainer_copy.device
    
    model_copy.to(device)
    trainer_copy.val_loader = trainer_copy.train_loader
    trainer_copy.epochs = 1
    trainer_copy.state.epoch = 0
    test = TestModelAccuracy(trainer_copy)
    results = test_results_to_score(test())

    model_copy.to("cpu")
    os.makedirs(model_path, exist_ok=True)
    tokenizer.save_pretrained(os.path.join(model_path, 'tokenizer'))
    torch.save({
        'model': model_copy.state_dict(),
        'hparams': model_copy.hparams}, os.path.join(model_path, 'model'))
    return results
