"""Abstract test classes"""

# pylint: disable=lost-exception

from abc import ABC, abstractmethod
import traceback


class bcolors:
    
    COLORS = {"blue": "\033[94m", "green": "\033[92m", "red": "\033[91m", "cyan": "\033[96m", "yellow": "\033[93m"}
    
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    
    @staticmethod
    def colorize(color, string):
        return f"{bcolors.COLORS[color]}{string}{bcolors.ENDC}"
    
    @staticmethod
    def underline(color, string):
        return f"{bcolors.COLORS[color]}{bcolors.UNDERLINE}{string}{bcolors.ENDC}"
    
    @staticmethod
    def failure_msg():
        return f"{bcolors.colorize('red', 'failed')}"
    
    @staticmethod
    def success_msg():
        return f"{bcolors.colorize('green', 'passed!')}"
    
    @staticmethod
    def colorful_scores(score, max_score):
        score = bcolors.colorize('green', str(score)) if score == max_score else bcolors.colorize('red', str(score))
        return f"{score}/{bcolors.colorize('green', str(max_score))}"

class string_utils:
    
    HASHTAGS = "#######"
    TEST_START = f"\n{HASHTAGS} Testing Started {HASHTAGS}\n"
    TEST_FINISHED = f"\n{HASHTAGS} Testing Finished {HASHTAGS}"
    ARROW = " --> "
    NEWLINE = "\n"
    EMPTY = ""
    
    @staticmethod
    def print_test_start(test_name=""):
        print(f"\n{string_utils.HASHTAGS} Testing {bcolors.colorize('cyan', test_name)} Started {string_utils.HASHTAGS}\n")
        
    @staticmethod
    def print_test_finished(test_name=""):
        print(f"\n{string_utils.HASHTAGS} Testing {bcolors.colorize('cyan', test_name)} Finished {string_utils.HASHTAGS}")


class UnitTest(ABC):
    """
    Abstract class for a single test
    All subclasses have to overwrite test() and failure_message()
    Then the execution order is the following:
        1. test() method is executed
        2. if test() method returned False or threw an exception,
            print the failure message defined by failure_message()
        3.  return a tuple (tests_failed, total_tests)
    """
    
    def __init__(self, *args, **kwargs):
        
        self.define_name()
        self.test_name = f"Test {self.name}:"
        self.failed_msg = bcolors.failure_msg()
        self.success_msg = bcolors.success_msg()

    def __call__(self):
        try:
            test_passed = self.test()
            if test_passed:
                print(self.define_success_message())
                return 0, 1  # 0 tests failed, 1 total test
            print(self.define_failure_message())
            return 1, 1  # 1 test failed, 1 total test
        except Exception as exception:
            print(self.define_exception_message(exception))
            return 1, 1  # 1 test failed, 1 total test

    @abstractmethod
    def test(self):
        """Run the test and return True if passed else False"""

    def define_failure_message(self):
        """Define the message that should be printed upon test failure"""
        return f"{self.test_name} {bcolors.failure_msg()}"

    def define_success_message(self):
        """Define the message that should be printed upon test success"""
        return f"{self.test_name} {bcolors.success_msg()}"
    
    def define_exception_message(self, exception):
        """
        Define the message that should be printed if an exception occurs
        :param exception: exception that was thrown
        """
        return self.emphsize(f"{self.test_name} {bcolors.failure_msg()} with exception: \n\n{traceback.format_exc()}")
    
    def emphsize(self, string):
        hashtag = bcolors.colorize('yellow', string_utils.HASHTAGS)
        return f"\n{hashtag}\n{string}{hashtag}\n"
    
    def define_name(self):
        """Define the name of the test"""
        self.name = type(self).__name__


class CompositeTest(ABC):
    """
    Abstract class for a test consisting of multiple other tests
    All subclasses have to overwrite define_tests(), success_message(),
    and failure_message().
    Then the execution order is the following:
    1. run all tests
    2. if all tests passed, print success message
    3. if some tests failed, print failure message
         and how many tests passed vs total tests
    4. return a tuple (tests_failed, total_tests)
    """
    def __init__(self, *args, **kwargs):
        self.tests = self.define_tests(*args, **kwargs)
        self.name = type(self).__name__
        self.test_name = f"Test {self.name}:"

    @abstractmethod
    def define_tests(self, *args, **kwargs):
        """Define a list of all sub-tests that should be run"""

    def define_failure_message(self):
        """Define the message that should be printed upon test failure"""
        return f"{self.test_name} {bcolors.failure_msg()}"

    def define_success_message(self):
        """Define the message that should be printed upon test success"""
        return f"{self.test_name} {bcolors.success_msg()}"

    def __call__(self):
        tests_failed, tests_total = 0, 0
        
        string_utils.print_test_start(self.name)
        for test in self.tests:
            new_fail, new_total = test()
            tests_failed += new_fail
            tests_total += new_total
            
        tests_passed = tests_total - tests_failed
        
        string_utils.print_test_finished(self.name)
        if tests_failed == 0:
            print(
                self.define_success_message() + string_utils.ARROW,
                f"Tests passed: {bcolors.colorful_scores(tests_passed, tests_total)}"
            )
        else:
            print(
                self.define_failure_message() + string_utils.ARROW,
                f"Tests passed: {bcolors.colorful_scores(tests_passed, tests_total)}"
            )
        return tests_failed, tests_total


class MethodTest(CompositeTest, ABC):
    """
    Abstract class to test methods using multiple tests
    Similar behaviour to CompositeTest, except that subclasses have to
    overwrite define_method_name instead of success_message and failure_message
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.method_name = self.define_method_name()

    @abstractmethod
    def define_method_name(self):
        """Define name of the method to be tested"""

    def define_success_message(self):
        return f"Method {self.method_name}(): {bcolors.success_msg()}"

    def define_failure_message(self):
        return f"Method {self.method_name}(): {bcolors.failure_msg()}"
    

class ConditionedMethodTest(CompositeTest, ABC):
    """
    Abstract class to test methods using multiple tests using a condition string
    Similar behaviour to CompositeTest, except that subclasses have to
    overwrite define_method_name instead of success_message and failure_message
    """
    def __init__(self, condition_string, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.method_name = self.define_method_name()
        self.condition_string = condition_string

    @abstractmethod
    def define_method_name(self):
        """Define name of the method to be tested"""
        
    def define_success_message(self):
        return f"Method {self.method_name}() (using {self.condition_string}): {bcolors.success_msg()}"

    def define_failure_message(self):
        return f"Method {self.method_name}() (using {self.condition_string}): {bcolors.failure_msg()}"

    
    def __call__(self):
        tests_failed, tests_total = 0, 0
        
        print(" ".join(f"{bcolors.underline('yellow', f'Testing {self.method_name}()')} with condition: \
            {bcolors.colorize('blue',self.condition_string)}. No. of test cases: {len(self.tests)}".split()))
        
        for test in self.tests:
            new_fail, new_total = test()
            tests_failed += new_fail
            tests_total += new_total
        tests_passed = tests_total - tests_failed
        if tests_failed == 0:
            print(
                self.define_success_message() + string_utils.ARROW,
                f"Tests passed: {bcolors.colorful_scores(tests_passed, tests_total)}"
            )
        else:
            print(
                self.define_failure_message() + string_utils.ARROW,
                f"Tests passed: {bcolors.colorful_scores(tests_passed, tests_total)}"
            )
        print(string_utils.EMPTY)
        return tests_failed, tests_total


class ClassTest(CompositeTest, ABC):
    """
    Abstract class to test classes using multiple tests
    Similar behaviour to CompositeTest, except that subclasses have to
    overwrite define_class_name instead of success_message and failure_message
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_name = self.define_class_name()

    @abstractmethod
    def define_class_name(self):
        """Define name of the class to be tested"""

    def define_success_message(self):
        return f"Class {self.class_name}: {bcolors.success_msg()}"

    def define_failure_message(self):
        return f"Class {self.class_name}: {bcolors.failure_msg()}"


def test_results_to_score(test_results, verbose=True):
    """Calculate a score from 0-100 based on number of failed/total tests"""
    tests_failed, tests_total = test_results
    tests_passed = tests_total - tests_failed
    score = int(100 * tests_passed / tests_total)
    if verbose:
        print(f"Score: {bcolors.colorful_scores(score, max_score=100)}")
    return score
