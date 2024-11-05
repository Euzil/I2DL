"""Tests for __len__() methods"""

from .base_tests import UnitTest, MethodTest, ConditionedMethodTest, string_utils


class LenTestInt(UnitTest):
    """Test whether __len__() method of an object returns type int"""
    def __init__(self, object_):
        super().__init__()
        self.object = object_

    def test(self):
        return isinstance(len(self.object), int)

    def define_failure_message(self):
        received_type = str(type(len(self.object)))
        return " ".join(f"{self.test_name} {self.failed_msg} {string_utils.ARROW}\
            Length is not of type int, got type {received_type}.".split())

class LenTestCorrect(UnitTest):
    """Test whether __len__() method of an object returns correct value"""
    def __init__(self, object_, len_):
        super().__init__()
        self.object = object_
        self.ref_len = len_

    def test(self):
        return len(self.object) == self.ref_len

    def define_failure_message(self):
        return " ".join(f"{self.test_name} {self.failed_msg} {string_utils.ARROW}\
            Length is incorrect (expected {self.ref_len}, got {len(self.object)}).".split()) 


class LenTest(MethodTest):
    """Test whether __len__() method of an object is correctly implemented"""
    
    def define_tests(self, object_, len_):
        return [LenTestInt(object_), LenTestCorrect(object_, len_)]

    def define_method_name(self):
        return "__len__"

    
class ConditionedLenTest(ConditionedMethodTest):
    """Test whether __len__() method of an object is correctly implemented using a condition"""
    def __init__(self, condition_string, *args, **kwargs):
        super().__init__(condition_string, *args, **kwargs)
        
    def define_tests(self, object_, len_):
        return [LenTestInt(object_), LenTestCorrect(object_, len_)]

    def define_method_name(self):
        return "__len__"
