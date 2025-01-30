import itertools
import torch
from torch.utils.data import IterableDataset
import pandas as pd
import os


def compute_length(file_paths):
    """
    Loops through all files and counts the number of lines

    Args:
        file_paths: List of files to loop through
    """
    length = 0
    line_count = 0
    for file in file_paths:
        with open(file, 'r') as f:
            for line_count, _ in enumerate(f):
                pass
        length += line_count
    return line_count


class CustomIterableDataset(IterableDataset):

    def __init__(self, file_paths, chunk_size: int = 10):
        """

        Args:
            file_paths: List of files to loop through (Also accepts single file)
            chunk_size: Number of sentences to load into memory at a time (default=10)

        Attributes:
            self.length: Length of entire dataset
        """
        if type(file_paths) is not list:
            file_paths = [file_paths]
        self.file_paths = file_paths
        self.chunk_size = chunk_size
        self.length = compute_length(self.file_paths)

    def __len__(self):
        """
        Returns the length of the Dataset
        """
        return self.length

    def parse_file(self):
        """
        Parses the files in chunks and yields 'source' and 'target' data.

        Yields:
            dict: Contains 'source' and 'target' data from the file.
        """
        for file in self.file_paths:
            reader = pd.read_csv(filepath_or_buffer=file, iterator=True, chunksize=self.chunk_size)

            ########################################################################
            # TODO:                                                                #
            #   Task 1:                                                            #
            #       - Loop through all chunks in reader                            #
            #       - Loop through all rows in chunk                               #
            #       - 'return' a dictionary: {'source': source_data,               #
            #                                 'target': target_data}               #
            # Hints:                                                               #
            #       - Use iterrows() to iterate through all rows! Have a look at   #
            #         at pandas implementation of this function and see what it    #
            #         returns!                                                     #
            #       - The dataframe we are reading in this case has two columns:   #
            #         'source' and 'target'. You can index them using something    #
            #         like row['source'].                                          #
            #         Dont use return ;)                                           #
            ########################################################################


            for chunk in reader:
                for _, row in chunk.iterrows():
                    yield {
                        'source': row['source'],
                        'target': row['target']
                    }

            ########################################################################
            #                           END OF YOUR CODE                           #
            ########################################################################

    def __iter__(self):
        """
        Iterates through the dataset, managing parallelism when using multiple workers.

        Returns:
            iterator: Iterator over the dataset, considering parallel processing.
        """
        os.environ["TOKENIZERS_PARALLELISM"] = "True"
        iterator = self.parse_file()
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            os.environ["TOKENIZERS_PARALLELISM"] = "False"
            worker_total_num = worker_info.num_workers
            worker_id = torch.utils.data.get_worker_info().id
            return itertools.islice(iterator, worker_id, None, worker_total_num)
        return iterator
