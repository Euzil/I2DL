class AttentionScoresSaver:
    """
    Module to save and visualize scores
    """

    def __init__(self):
        self._save_scores = False
        self._scores = []

    def save(self, score):
        if self._save_scores:
            self._scores.append(score)

    def record_scores(self):
        self._save_scores = True

    def reset(self):
        self._scores = []
        self._save_scores = False

    def get_scores(self, reset=True):
        scores = self._scores
        self.reset()
        return scores
    
SCORE_SAVER = AttentionScoresSaver()

from .attention import ScaledDotAttention
from .embedding import positional_encoding
from .embedding import Embedding
from .multi_head_attention import MultiHeadAttention



