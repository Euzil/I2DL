"""Tests for Transform classes in data/image_folder_dataset.py"""

import numpy as np
from tqdm import tqdm
import re

from .base_tests import UnitTest, MethodTest, ClassTest, test_results_to_score, string_utils


class TransformUnitTest(UnitTest):
    
    def __init__(self, orig_dataset, student_transfromed_dataset):
        super().__init__()
        self.orig_dataset = orig_dataset
        self.student_transfromed_dataset = student_transfromed_dataset
        self.num_candidates = 10
        self.epsilon = 1e-5
        self.func = lambda x: x
        self.error = None
        self.error_index = 0
        
    def rescale(self, image, image_range=(0, 255), rescaled_range=(0, 1)):
        """Rescale image to range [0, 1]"""
        image_min, image_max = image_range
        rescaled_min, rescaled_max = rescaled_range
        image = (image - image_min) / (image_max - image_min)
        image = image * (rescaled_max - rescaled_min) + rescaled_min
        return image
    
    def normalize(self, image, mean, std):
        """Normalize image"""
        image = (image - mean) / std
        return image 
    
    def test(self):
        """Test whether transform is applied correctly"""
        for i in range(self.num_candidates):
            student_image = self.student_transfromed_dataset[i]["image"]
            orig_image = self.orig_dataset[i]["image"]
            orig_image = self.func(orig_image)
            if np.any(np.abs(student_image - orig_image > self.epsilon)):
                self.error = np.abs(np.mean(student_image - orig_image))
                self.error_index = i
                return False
        return True
    
    def define_failure_message(self):
        return " ".join(f"{self.test_name} {self.failed_msg} {string_utils.ARROW}\
            Transform is not applied correctly (mean error of {self.error} at index {self.error_index}).".split())
    

class RescaleTransformUnitTest(TransformUnitTest):
    
    def __init__(self, orig_dataset, student_rescaled_dataset, image_range=(0, 255), rescaled_range=(0, 1)):
        super().__init__(orig_dataset, student_rescaled_dataset)
        
        self.image_range = image_range
        self.rescaled_range = rescaled_range
        self.func = lambda x: self.rescale(x, self.image_range, self.rescaled_range)


class RescaleTransformTest(ClassTest):
    """Test class RescaleTransform"""
    def define_tests(self, orig_dataset, student_rescaled_dataset, image_range=(0, 255), rescaled_range=(0, 1)):
        return [
            RescaleTransformUnitTest(orig_dataset, student_rescaled_dataset, image_range, rescaled_range),
        ]

    def define_class_name(self):
        return "RescaleTransform"


def test_rescale_transform(orig_dataset, student_rescaled_dataset, image_range=(0, 255), rescaled_range=(0, 1)):
    """Test class RescaleTransform"""
    test = RescaleTransformTest(orig_dataset, student_rescaled_dataset, image_range, rescaled_range)
    return test_results_to_score(test())


class CIFARImageStatisticTest(UnitTest):
    """Test whether computed CIFAR-10 image std is correct """
    def __init__(self, values, expected_values, variable="mean"):
        super().__init__()
        
        self.values = np.array(values)
        self.expected_values = np.array(expected_values)
        self.epsilon = 1e-5
        self.vairable = variable
        self.test_name = f"Test {type(self).__name__} ({self.vairable}):"

    def test(self):
        
        assert self.vairable in ["mean", "std"], "variable should be either 'mean' or 'std'"
        assert np.iterable(self.values), "values should be an array of shape (3,)"
        assert type(self.values) == np.ndarray, "values should be a numpy array"
    
        return np.all(np.abs((self.values - self.expected_values)) < self.epsilon) and self.values.shape == (3,)

    def define_failure_message(self):
        
        if self.values.shape != (3,):
            return " ".join(f"{self.test_name} {self.failed_msg} {string_utils.ARROW} Computed image {self.vairable} shape is incorrect. \
            Hint: the result array should be of shape (1x3). You may use Numpy's built-in functions.".split())
        else:
            error = np.mean(np.abs(self.values - self.expected_values))
            return " ".join(f"{self.test_name} {self.failed_msg} {string_utils.ARROW} Computed image {self.vairable} values are incorrect. \
            Hint: The mean error is {error}. You may use Numpy's built-in functions.".split())


class CIFARImageMeanStdTest(MethodTest):
    """Test compute_image_mean_and_std() in data/image_folder_dataset.py"""

    def define_tests(self, mean, std):
        return [
            CIFARImageStatisticTest(mean, expected_values=np.array([0.49191375, 0.48235852, 0.44673872]), variable="mean"),
            CIFARImageStatisticTest(std, expected_values=np.array([0.24706447, 0.24346213, 0.26147554]), variable="std"),
        ]

    def define_method_name(self):
        return "compute_image_mean_and_std"


def test_compute_image_mean_and_std(mean, std):
    """Test compute_image_mean_and_std() in data/image_folder_dataset.py"""
    test = CIFARImageMeanStdTest(mean, std)
    return test_results_to_score(test())


class NormalizationTest(TransformUnitTest):
    """Test whether NormalizationTransform normalizes correctly"""
    def __init__(self, dataset, student_dataset, mean, std):
        super().__init__(dataset, student_dataset)
        self.func = lambda x: self.normalize(self.rescale(x), mean, std)

class NormalizationTransformTest(MethodTest):
    """Test class NormalizationTransform"""
    def define_tests(self, orig_data, dataset, cifar_mean, cifar_std):
        return [
            NormalizationTest(orig_data, dataset, cifar_mean, cifar_std),
        ]

    def define_method_name(self):
        return "NormalizationTransform"

def test_normalization_transform(orig_data, dataset, cifar_mean, cifar_std):
    """Test class NormalizationTransform"""
    test = NormalizationTransformTest(orig_data, dataset, cifar_mean, cifar_std)
    return test_results_to_score(test())