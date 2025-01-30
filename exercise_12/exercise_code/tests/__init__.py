"""Define tests, sanity checks, and evaluation"""
import os
tensor_path = os.path.join(os.getcwd(), 'exercise_code', 'tests', 'tensors')

from .iterable_dataset_test import test_task_1
from .attention_test import test_task_6
from .feed_forward_test import test_task_3
from .encoder_test import test_task_4, test_task_5
from .decoder_test import test_task_7, test_task_8
from .transformer_test import test_task_9, test_task_10, test_task_11, test_model_parameters, test_and_save_model


