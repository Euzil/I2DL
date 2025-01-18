"""Define tests, sanity checks, and evaluation"""
import os
tensor_path = os.path.join(os.getcwd(), 'exercise_code', 'tests', 'tensors')

from .embedding_test import test_task_1, test_task_4
from .attention_test import test_task_2, test_task_3


