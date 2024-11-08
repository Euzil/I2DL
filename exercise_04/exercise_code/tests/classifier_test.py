from .base_tests import UnitTest, CompositeTest, MethodTest, test_results_to_score, string_utils
import numpy as np
import math

def rel_error(x, y):
    """ returns relative error """
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


class Sigmoid_Of_Zero(UnitTest):
    """Test whether Sigmoid of 0 is correct"""

    def __init__(self, Classifier):
        super().__init__()
        self.value = Classifier.sigmoid(float(0))
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
        self.value = Classifier.sigmoid(float(100))
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


class ClassifierForwardTest(UnitTest):
    def __init__(self, Classifier):
        super().__init__()
        
        Classifier.initialize_weights()
        sample_x = np.array([1, 2,
                             3, 4]).reshape(2, 2)
        self.value = Classifier.forward(sample_x)
        
        sample_x = np.concatenate((sample_x, np.ones((2, 1))), axis=1)
        self.truth = Classifier.sigmoid(sample_x.dot(Classifier.W))
        
        

    def test(self):

        return (rel_error(self.truth, self.value) < 1e-6)

    def define_failure_message(self):
        return " ".join(f"{self.test_name} {self.failed_msg} {string_utils.ARROW}\
            Classifier forward is incorrect. Expected {self.truth}, but evaluated\
                {self.value}".split()) 
   
class ClassifierSigmoidTest(UnitTest):
    def __init__(self, Classifier):
        super().__init__()
        sample_x = np.array([1, 2,
                             3, 4]).reshape(2, 2)
        self.value = Classifier.sigmoid(sample_x)
        self.truth = 1 / (1 + np.exp(-sample_x))

    def test(self):
        return (rel_error(self.truth, self.value) < 1e-6)

    def define_failure_message(self):
        return " ".join(f"{self.test_name} {self.failed_msg} {string_utils.ARROW}\
            Sigmoid forward is incorrect. Expected {self.truth}, but evaluated\
                {self.value}".split()) 
       


class ClassifierBackwardTest(UnitTest):
    def __init__(self, Classifier):
        super().__init__()
        Classifier.initialize_weights()
        sample_x = np.array([1, 2,
                             3, 4]).reshape(2, 2)
        sample_y = Classifier.forward(sample_x)
        self.value = Classifier.backward(sample_y)
        self.truth = np.concatenate(
            (sample_x, np.ones((2, 1))), axis=1).T.dot(sample_y * (1 - sample_y) * sample_y)

    def test(self):
        return (rel_error(self.truth, self.value) < 1e-6)

    def define_failure_message(self):
        return " ".join(f"{self.test_name} {self.failed_msg} {string_utils.ARROW}\
            Classifier backward is incorrect. Expected {self.truth}, but evaluated\
                {self.value}".split()) 


class SigmoidMethodTest(MethodTest):
    def define_tests(self, Classifier):
        return [Sigmoid_Of_Zero(Classifier),
                Sigmoid_Of_Zero_Array(Classifier),
                Sigmoid_Of_100(Classifier),
                Sigmoid_Of_Array_of_100(Classifier)
                ]

    def define_method_name(self):
        return "sigmoid"


class ForwardMethodTest(MethodTest):
    def define_tests(self, Classifier):
        return [ClassifierForwardTest(Classifier)
                ]

    def define_method_name(self):
        return "forward"


class BackwardMethodTest(MethodTest):
    def define_tests(self, Classifier):
        return [ClassifierBackwardTest(Classifier)
                ]

    def define_method_name(self):
        return "backward"


class ClassifierTest(CompositeTest):
    def define_tests(self, Classifier):
        return [SigmoidMethodTest(Classifier),
                ForwardMethodTest(Classifier),
                BackwardMethodTest(Classifier)
                ]


def test_classifier(Classifier):
    """Test the Classifier"""
    test = ClassifierTest(Classifier)
    return test_results_to_score(test())
