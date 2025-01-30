from .base_tests import UnitTest, string_utils, test_results_to_score, CompositeTest
import pandas as pd


class IterableDatasetKeyTest(UnitTest):

    def __init__(self):
        super().__init__()

        from ..data import CustomIterableDataset

        file_path = "../datasets/transformerDatasets/dummyDatasets/ds_dummy"
        dataset = CustomIterableDataset(file_path)
        iterable = dataset.__iter__()

        self.dictionary = next(iterable)

    def test(self):
        result = set(self.dictionary.keys())
        expected = {'target', 'source'}

        return result == expected

    def define_failure_message(self):
        result = set(self.dictionary.keys())
        expected = {'target', 'source'}
        return " ".join(f"{self.test_name} {self.failed_msg} {string_utils.ARROW}\
            Expected keys {expected}, got {result}.".split())


class IterableDatasetValueTest(UnitTest):
    def __init__(self):
        super().__init__()

        from ..data import CustomIterableDataset
        
        file_path = "../datasets/transformerDatasets/dummyDatasets/ds_dummy"
        dataset = CustomIterableDataset(file_path)
        iterable = dataset.__iter__()

        self.dictionary = next(iterable)
        self.dataframe = pd.read_csv(file_path)

    def test(self):
        result_source = self.dictionary['source']
        result_target = self.dictionary['target']
        expected_source = self.dataframe.iloc[0]['source']
        expected_target = self.dataframe.iloc[0]['target']
        return result_source == expected_source and result_target == expected_target

    def define_failure_message(self):
        return " ".join(f"{self.test_name} {self.failed_msg} {string_utils.ARROW}\
            Expected values do not match the correct keys.".split())


class TestTask1(CompositeTest):
    def define_tests(self, ):
        return [
            IterableDatasetKeyTest(),
            IterableDatasetValueTest()
        ]


def test_task_1():
    test = TestTask1()
    return test_results_to_score(test())
