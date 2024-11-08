import numpy as np
from .base_tests import UnitTest, CompositeTest, test_results_to_score, string_utils
import math


def rel_error(x, y):
    """ returns relative error """
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

class Sigmoid_Of_Zero(UnitTest):
    """Test whether Sigmoid of 0 is correct"""

    def __init__(self, Classifier):
        super().__init__()
        self.value = Classifier.sigmoid(np.float(0))
        self.truth = 0.5

    def test(self):
        return (rel_error(self.truth, self.value) < 1e-6)

    def define_failure_message(self):
        return " ".join(f"{self.test_name} {self.failed_msg} {string_utils.ARROW}\
            Expected Sigmoid(0) to be {self.truth}, \
                but evaluated {self.value}".split()) 
        


class Sigmoid_Of_Zero_Array(UnitTest):
    """Test whether Sigmoid of a numpy array [0, 0, 0, 0, 0] is correct"""

    def __init__(self, Classifier):
        super().__init__()
        self.value = Classifier.sigmoid(np.asarray([0, 0, 0, 0, 0]))
        self.truth = np.array([0.5, 0.5, 0.5, 0.5, 0.5])

    def test(self):
        return (rel_error(self.truth, self.value) < 1e-6)

    def define_failure_message(self):
        return " ".join(f"{self.test_name} {self.failed_msg} {string_utils.ARROW}\
            Expected Sigmoid([0,..,0]) to be {self.truth}, \
                but evaluated {self.value}".split()) 


class Sigmoid_Of_100(UnitTest):
    """Test whether Sigmoid of 100 is correct"""

    def __init__(self, Classifier):
        super().__init__()
        self.value = Classifier.sigmoid(np.float(100))
        self.truth = 1.0

    def test(self):
        return (rel_error(self.truth, self.value) < 1e-6)

    def define_failure_message(self):
        return " ".join(f"{self.test_name} {self.failed_msg} {string_utils.ARROW}\
            Expected Sigmoid(100) to be {self.truth}, \
                but evaluated {self.value}".split()) 


class Sigmoid_Of_Array_of_100(UnitTest):
    """Test whether Sigmoid of [100, 100, 100, 100, 100] is correct"""

    def __init__(self, Classifier):
        super().__init__()
        self.value = Classifier.sigmoid(np.asarray([100, 100, 100, 100, 100]))
        self.truth = np.array([1.0, 1.0, 1.0, 1.0, 1.0])

    def test(self):
        return (rel_error(self.truth, self.value) < 1e-6)

    def define_failure_message(self):
        return " ".join(f"{self.test_name} {self.failed_msg} {string_utils.ARROW}\
            Expected Sigmoid([100,...,100]) to be {self.truth}, \
                but evaluated {self.value}".split()) 


class RunAllSigmoidTests(CompositeTest):

    def define_tests(self, model):
        return [
            Sigmoid_Of_Zero(model),
            Sigmoid_Of_Zero_Array(model),
            Sigmoid_Of_100(model),
            Sigmoid_Of_Array_of_100(model),
        ]



class SigmoidTestWrapper:
    def __init__(self, model):
        self.sigmoid_tests = RunAllSigmoidTests(model)

    def __call__(self, *args, **kwargs):
        return str(test_results_to_score(self.sigmoid_tests()))

            