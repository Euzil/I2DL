"""
Definition of ImageFolderDataset dataset class
"""

# pylint: disable=too-few-public-methods

import os
import pickle

import numpy as np
from PIL import Image

from .base_dataset import Dataset
from .transforms import IdentityTransform


class ImageFolderDataset(Dataset):
    """CIFAR-10 dataset class"""
    def __init__(self, *args,
                 transform=IdentityTransform(),
                 download_url="https://i2dl.vc.in.tum.de/static/data/cifar10.zip",
                 **kwargs):
        super().__init__(*args, 
                         download_url=download_url,
                         **kwargs)
        
        self.classes, self.class_to_idx = self._find_classes(self.root_path)
        self.images, self.labels = self.make_dataset(
            directory=self.root_path,
            class_to_idx=self.class_to_idx
        )
        # transform function that we will apply later for data preprocessing
      
        self.transform = transform

    @staticmethod
    def _find_classes(directory):
        """
        Finds the class folders in a dataset
        :param directory: root directory of the dataset
        :returns: (classes, class_to_idx), where
          - classes is the list of all classes found
          - class_to_idx is a dict that maps class to label
        """
        classes = [d.name for d in os.scandir(directory) if d.is_dir()]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    @staticmethod
    def make_dataset(directory, class_to_idx):
        """
        Create the image dataset by preparaing a list of samples
        Images are sorted in an ascending order by class and file name
        :param directory: root directory of the dataset
        :param class_to_idx: A dict that maps classes to labels
        :returns: (images, labels) where:
            - images is a list containing paths to all images in the dataset, NOT the actual images
            - labels is a list containing one label per image
        """
        images, labels = [], []

        for target_class in sorted(class_to_idx.keys()):
            label = class_to_idx[target_class]
            target_dir = os.path.join(directory, target_class)
            for root, _, fnames in sorted(os.walk(target_dir)):
                for fname in sorted(fnames):
                    if fname.endswith(".png"):
                        path = os.path.join(root, fname)
                        images.append(path)
                        labels.append(label)

        assert len(images) == len(labels)
        return images, labels

    def __len__(self):
        length = None
        ########################################################################
        # TODO:                                                                #
        # Return the length of the dataset (number of images)                  #
        ########################################################################
        length = len(self.images)

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return length

    @staticmethod
    def load_image_as_numpy(image_path):
        """Load image from image_path as numpy array"""
        return np.asarray(Image.open(image_path), dtype=float)

    def __getitem__(self, index):
        data_dict = None
        ########################################################################
        # TODO:                                                                #
        # Create a dict of the data at the given index in your dataset         #
        # The dict should be of the following format:                          #
        # {"image": <i-th image>,                                              #
        # "label": <label of i-th image>}                                      #
        #                                                                      #
        # Hint 1:                                                              #
        #   use self.load_image_as_numpy() to load an image from a             #
        #   file path. Note: you have to use "self.load_image_as_numpy()"      #
        #   and not "ImageFolderDataset.load_image_as_numpy()", as it will     #
        #  cause a bug, when using  MemoryImageFolderDataset                   #
        #                                                                      # 
        # Hint 2:                                                              #
        #   If applicable (Task 4: 'Transforms and Image Preprocessing'),      #
        #   make sure to apply self.transform to the image if self.transform   #
        #   is defined (not None):                                             #
        #               image_transformed = self.transform(image)              #
        #                                                                      #
        # Hint 3:                                                              #
        #   The labels are supposed to be numbers, in the range of [0, 9],     #
        #   not strings.                                                       #
        #                                                                      #    
        # Hint 4: the labels and images are already prepared and stored in     #
        #  self.labels and self.images. DO NOT call self.make_dataset() again! #    
        ########################################################################
   
        '''
         # Step 1: 获取图像路径和标签
        image_path = self.images[index]
        label = self.labels[index]

        # Step 2: 加载图像
        image = self.load_image_as_numpy(image_path)

        # Step 3: 如果 self.transform 存在，应用转换
        if self.transform is not None:
            image = self.transform(image)

        # Step 4: 创建数据字典
        data_dict = {
            "image": image,
            "label": label
        }
        '''

        '''
         # 获取图像路径和对应标签
        image_path = self.images[index]
        label = self.labels[index]

        # 使用 self.load_image_as_numpy() 加载图像
        image = self.load_image_as_numpy(image_path)

        # 如果 self.transform 被定义，则应用转换
        if self.transform is not None:
            image = self.transform(image)

        # 返回字典格式的数据项
        data_dict = {
            "image": image,
            "label": label
        }
        '''
        data_dict = {}

        # 加载指定索引的图像和标签
        image_path = self.images[index]
        label = self.labels[index]

        # 使用自定义的加载方法加载图像（例如，从文件路径加载为 numpy 数组）
        image = self.load_image_as_numpy(image_path)

        # 如果定义了转换，则对图像应用转换
        if self.transform is not None:
            image = self.transform(image)

        # 将图像和标签存入字典
        data_dict["image"] = image
        data_dict["label"] = label



        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return data_dict


class MemoryImageFolderDataset(ImageFolderDataset):
    def __init__(self, root, *args,
                 transform=IdentityTransform(),
                 download_url="https://i2dl.vc.in.tum.de/static/data/cifar10memory.zip",
                 **kwargs):
        # Fix the root directory automatically
        if not root.endswith('memory'):
            root += 'memory'

        super().__init__(
            root, *args, download_url=download_url, **kwargs)
        
        with open(os.path.join(
            self.root_path, 'cifar10.pckl'
            ), 'rb') as f:
            save_dict = pickle.load(f)

        self.images = save_dict['images']
        self.labels = save_dict['labels']
        self.class_to_idx = save_dict['class_to_idx']
        self.classes = save_dict['classes']

        self.transform = transform

    def load_image_as_numpy(self, image_path):
        """Here we already have everything in memory,
        so we can just return the image"""
        return image_path

        