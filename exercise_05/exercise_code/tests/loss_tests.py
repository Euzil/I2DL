from .base_tests import UnitTest, CompositeTest, MethodTest, test_results_to_score, string_utils
import numpy as np
import math
# from exercise_code.networks.loss import *


def eval_numerical_gradient(f, x, h=1e-6):
    """
    a naive implementation of numerical gradient of f at x
    - f should be a function that takes a single argument
    - x is the point (numpy array) to evaluate the gradient at
    """
    grad = (f(x + h) - f(x - h)) / (2 * h)
    grad /= len(x) # In order to compare it with the normalized graident.
    return grad

def rel_error(x, y):
    """ returns relative error """
    
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

class LossForwardUnitTest(UnitTest):
    def __init__(self, studnet_loss, inputs, targets):
        super().__init__()
        self.student_loss_func = studnet_loss
        self.inputs = inputs
        self.targets = targets
        
        self.student_value = self.student_loss_func(self.inputs, self.targets)
        self.expected_result = self.ground_truth()
        self.name = "LossFuncUnitTest"
        self.eps = 1e-5
        
    def ground_truth(self):
        """Ground truth implementation of loss function"""
        raise NotImplementedError("Ground truth implementation of loss function is not implemented")
        
    def test(self):
        return ((self.student_value - self.expected_result) < self.eps).all()

    def define_failure_message(self):
        return " ".join(f"{self.test_name} {self.failed_msg} {string_utils.ARROW}\
            {self.name} is incorrect. Expected {self.expected_result}, \
                but got {self.student_value}".split())
    
    
class LossBackwardNormalUnitTest(UnitTest):
    
    def __init__(self, loss, y_out, y_truth):
        super().__init__()
        self.y_out = y_out
        self.y_truth = y_truth
        self.student_grad = loss.backward(y_out, y_truth)
        self.loss = loss
        self.error = 0.0
        self.eps = 1e-8
        self.name = "LossBackwardNormalUnitTest"
        
    def ground_truth(self):
        
        def f(y):
            return self.loss(y, self.y_truth, individual_losses=True)
        
        return eval_numerical_gradient(f, np.array(self.y_out))
        
    def test(self):
        self.expected_grad = self.ground_truth()
        self.error = rel_error(self.expected_grad, self.student_grad)
        return self.error < self.eps

    def define_failure_message(self):
        return " ".join(f"{self.test_name} {self.failed_msg} {string_utils.ARROW}\
            {self.name} is incorrect. Expected error to be x < 1e-8, \
                but evaluated {self.error}".split())
            
class LossBackwardZeroUnitTest(UnitTest):
    
    def __init__(self, loss, y_out, y_truth, expected_gradient):
        super().__init__()
        self.y_out = y_out
        self.y_truth = y_truth
        self.expected_gradient = expected_gradient
        self.student_value = loss.backward(y_out, y_truth)
        self.name = "LossBackwardZeroUnitTest"
        
    def test(self):
        return (self.student_value == self.expected_gradient).all()
    
    def define_failure_message(self):
        return " ".join(f"{self.test_name} {self.failed_msg} {string_utils.ARROW}\
            {self.name} at x=0 is incorrect. Expected {self.expected_gradient}, \
                but evaluated {self.student_value}".split())
    

###################################### L1 ######################################
class L1ForwardTest(LossForwardUnitTest):
    
    def __init__(self, student_loss, inputs, targets):
        super().__init__(student_loss, inputs, targets)
        self.name = "L1ForwardTest"
    
    def ground_truth(self):
        return np.abs(self.inputs - self.targets)
    
class L1BackwardTestNormal(LossBackwardNormalUnitTest):
    
    def __init__(self, loss, y_out, y_truth):
        super().__init__(loss, y_out, y_truth)
        self.name = "L1 Backward"
    
class L1BackwardTestZero(LossBackwardZeroUnitTest):
    
    def __init__(self, loss, y_out, y_truth, expected_gradient):
        super().__init__(loss, y_out, y_truth, expected_gradient)
        self.name = "L1 Backward"
        
        
class L1BackwardTest(MethodTest):
    def define_tests(self, loss):
        return [L1BackwardTestNormal(loss, np.array([5., 2.]), np.array([3., 4.])),
                L1BackwardTestZero(loss, np.array([1., 1.]), np.array([1., 1.]), np.array([0., 0.]))]

    def define_method_name(self):
        return "L1.backward"
class L1Test(CompositeTest):
    def define_tests(self, loss):
        return [
            L1ForwardTest(loss, np.array([7., 2.]), np.array([5., 4.])),
            L1BackwardTestZero(loss, np.array([1., 1.]), np.array([1., 1.]), np.array([0., 0.])),
            L1BackwardTestNormal(loss, np.array([5., 2.]), np.array([3., 4.])),
        ]
        
def test_L1(loss):
    test = L1Test(loss)
    test_results_to_score(test())

###################################### L2 / MSE ######################################

class MSEForwardTest(LossForwardUnitTest):
    
    def __init__(self, student_loss, inputs, targets):
        super().__init__(student_loss, inputs, targets)
        self.name = "MSEForwardTest"
    
    def ground_truth(self):
        return (self.inputs - self.targets) ** 2
    
class MSEBackwardTestNormal(LossBackwardNormalUnitTest):
    
    def __init__(self, loss, y_out, y_truth):
        super().__init__(loss, y_out, y_truth)
        self.name = "MSE Backward"

class MSEBackwardTestZero(LossBackwardZeroUnitTest):
    
    def __init__(self, loss, y_out, y_truth, expected_gradient):
        super().__init__(loss, y_out, y_truth, expected_gradient)
        self.name = "MSE Backward"
 
class MSETestComposite(CompositeTest):
    def define_tests(self, loss):
        return [
            MSEForwardTest(loss, np.array([7., 2.]), np.array([5., 4.])),
            MSEBackwardTestNormal(loss, np.array([5., 2.]), np.array([3., 4.]))
        ]

def test_mse(loss):
    test = MSETestComposite(loss)
    test_results_to_score(test())


 
###################################### BCE ######################################

class BCEForwardTest(LossForwardUnitTest):
    
    def __init__(self, student_loss, inputs, targets):
        super().__init__(student_loss, inputs, targets)
        self.name = "BCEForwardTest"
    
    def ground_truth(self):
        likelihood = - (self.targets * np.log(self.inputs) + (1 - self.targets) * np.log(1 - self.inputs))
        return sum(likelihood) / len(likelihood)
    
class BCEBackwardTestNormal(LossBackwardNormalUnitTest):
    
    def __init__(self, loss, y_out, y_truth):
        super().__init__(loss, y_out, y_truth)
        self.name = "BCE Backward"
class BCETest(CompositeTest):
    
    def define_tests(self, loss):
        return [
            BCEForwardTest(loss, np.array([0.5, 0.3]), np.array([2., 4.])),
            BCEBackwardTestNormal(loss, np.array([0.5, 0.3]), np.array([2., 4.]))
        ]

def test_bce(loss):
    test = BCETest(loss)
    test_results_to_score(test())
