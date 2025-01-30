from .base_tests import UnitTest, string_utils, test_results_to_score, CompositeTest
import torch
from ..util.transformer_util import create_causal_mask
import numpy as np

class AttentionPaddingTest(UnitTest):
    def __init__(self):
        super().__init__()

        from ..network import ScaledDotAttention, SCORE_SAVER

        sequence_length = np.random.randint(low=30, high=50)
        d_k = np.random.randint(low=30, high=100)
        attention = ScaledDotAttention(d_k=d_k)
        random_input = torch.rand(size=(sequence_length, d_k))
        mask = create_causal_mask(sequence_length).squeeze(0)
        
        SCORE_SAVER.record_scores()
        attention(random_input, random_input, random_input, mask)
        scores = SCORE_SAVER.get_scores()[-1]
        self.result = scores * ~mask
        self.expected = torch.zeros_like(scores)

    def test(self):
        return torch.allclose(self.result, self.expected, atol=1e-2)

    def define_failure_message(self):
        return " ".join(f"{self.test_name} {self.failed_msg} {string_utils.ARROW}\
            Masked Softmax not implemented correctly.".split())
    
class MultiHeadAttentionPaddingTest(UnitTest):
    def __init__(self):
        super().__init__()

        from ..network import MultiHeadAttention, SCORE_SAVER

        sequence_length = np.random.randint(low=30, high=50)
        d_model = np.random.randint(low=30, high=100)
        d_k = np.random.randint(low=30, high=100)
        d_v = np.random.randint(low=30, high=100)
        n_heads = np.random.randint(low=1, high=10)
        multi_head_attention = MultiHeadAttention(d_model=d_model,
                                                       d_k=d_k,
                                                       d_v=d_v,
                                                       n_heads=n_heads)
        random_input = torch.rand(size=(5, sequence_length, d_model))
        mask = torch.broadcast_to(create_causal_mask(sequence_length), (5, sequence_length, sequence_length))
        
        SCORE_SAVER.record_scores()
        try:
            multi_head_attention(random_input, random_input, random_input, mask)
        except RuntimeError as e:
            if "The size of tensor" in str(e):
                print(f"Size mismatch error caught! This might be caused by not unsqueezing the mask correctly in multi_head_attention!")
                self.result = torch.tensor(0)
                self.expected = torch.tensor(1)
                return
            else:
                raise  # Re-raise the exception if it's not the size mismatch error
        scores = SCORE_SAVER.get_scores()[-1]
        self.result = scores * ~mask.unsqueeze(1)
        self.expected = torch.zeros_like(scores)

    def test(self):
        return torch.allclose(self.result, self.expected, atol=1e-2)

    def define_failure_message(self):
        return " ".join(f"{self.test_name} {self.failed_msg} {string_utils.ARROW}\
            Masked Softmax not implemented correctly.".split())


class TestTask6(CompositeTest):
    def define_tests(self):
        return [AttentionPaddingTest(),
                MultiHeadAttentionPaddingTest()]


def test_task_6():
    test = TestTask6()
    return test_results_to_score(test())
