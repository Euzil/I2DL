from .base_tests import UnitTest, CompositeTest, MethodTest, test_results_to_score, string_utils
import numpy as np
import math


def rel_error(x, y):
    """ returns relative error """
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


class SolverStepTest(UnitTest):
    """Test whether Solver._step() updates the model parameter correctly"""

    def __init__(self, Solver):
        super().__init__()
        Solver._step()
        self.truth = [[0.11574258], [0.0832162]]
        self.value = Solver.model.W
        pass

    def test(self):

        return (rel_error(self.truth, self.value) < 1e-6)

    def define_failure_message(self):
        return " ".join(f"{self.test_name} {self.failed_msg} {string_utils.ARROW}\
            The solver tep is incorrect. Expected {self.truth}, \
                but evaluated {self.value}".split()) 
        
class SolverTest(MethodTest):
    def define_tests(self, Solver):
        return [
            SolverStepTest(Solver)
        ]


    def define_method_name(self):
        return "_step"


def test_solver(Solver):
    """Test the Solver"""
    test = SolverTest(Solver)
    return test_results_to_score(test())
