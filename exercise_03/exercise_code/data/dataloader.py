"""Definition of Dataloader"""

import numpy as np


class DataLoader:
    """
    Dataloader Class
    Defines an iterable batch-sampler over a given dataset
    """
    def __init__(self, dataset, batch_size=1, shuffle=False, drop_last=False):
        """
        :param dataset: dataset from which to load the data
        :param batch_size: how many samples per batch to load
        :param shuffle: set to True to have the data reshuffled at every epoch
        :param drop_last: set to True to drop the last incomplete batch,
            if the dataset size is not divisible by the batch size.
            If False and the size of dataset is not divisible by the batch
            size, then the last batch will be smaller.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __iter__(self):
        ########################################################################
        # TODO:                                                                #
        # Define an iterable function that samples batches from the dataset.   #
        # Each batch should be a dict containing numpy arrays of length        #
        # batch_size (except for the last batch if drop_last=True)             #
        # Hints:                                                               #
        #   - np.random.permutation(n) can be used to get a list of all        #
        #     numbers from 0 to n-1 in a random order                          #
        #   - To load data efficiently, you should try to load only those      #
        #     samples from the dataset that are needed for the current batch.  #
        #     An easy way to do this is to build a generator with the yield    #
        #     keyword, see https://wiki.python.org/moin/Generators             #
        #   - Have a look at the "DataLoader" notebook first. This function is #
        #     supposed to combine the functions:                               #
        #       - combine_batch_dicts                                          #
        #       - batch_to_numpy                                               #
        #       - build_batch_iterator                                         #
        #     in section 1 of the notebook.                                    #
        ########################################################################
        

        indices = np.arange(len(self.dataset))  # 获取数据集的所有索引
        if self.shuffle:
            np.random.shuffle(indices)  # 随机打乱索引

        for start_idx in range(0, len(indices), self.batch_size):
            end_idx = start_idx + self.batch_size  # 计算当前批次的结束索引
            
            if self.drop_last and end_idx > len(indices):
                break  # 如果设置了 drop_last，且当前批次不完整，则结束

            batch_indices = indices[start_idx:end_idx]  # 获取当前批次的索引
            batch = {
                "data": []
            }

            for idx in batch_indices:
                sample = self.dataset[idx]  # 从数据集中加载样本
                
                # 添加错误处理以防止 KeyError
                if "data" not in sample:
                    raise KeyError(f"样本不包含所需的键: {sample.keys()}")

                data_value = sample["data"]

                # 确保加载的样本是唯一的
                if data_value not in batch["data"]:
                    batch["data"].append(data_value)  # 仅在未重复时添加

            if len(batch["data"]) == 0:
                continue  # 如果当前批次为空，则跳过

            # 将列表转换为 numpy 数组
            yield {key: np.array(value) for key, value in batch.items()}  # 使用 yield 返回当前批次

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def __len__(self):
        length = None
        ########################################################################
        # TODO:                                                                #
        # Return the length of the dataloader                                  #
        # Hint: this is the number of batches you can sample from the dataset. #
        # Don't forget to check for drop last (self.drop_last)!                #
        ########################################################################
        

        if self.drop_last:
            return len(self.dataset) // self.batch_size  # 整数除法
        else:
            return (len(self.dataset) + self.batch_size - 1) // self.batch_size  # 向上取整

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        # return length
