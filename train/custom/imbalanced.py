from typing import Callable

import pandas as pd
import torch
import torch.utils.data
import torchvision


class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices: a list of indices
        num_samples: number of samples to draw
        callback_get_label: a callback-like function which takes two arguments - dataset and index
    """

    def __init__(self, dataset, indices: list = None, num_samples: int = None, callback_get_label: Callable = None):
        # if indices is not provided, all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) if indices is None else indices

        # define custom callback
        self.callback_get_label = callback_get_label

        # if num_samples is not provided, draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) if num_samples is None else num_samples

        ###################################################################################
        labels = self._get_labels(dataset)
        
        # Example for object detection's labels
        # [ [1],
        #   [2, 4],
        #   [4, 5, 6],
        #   [3, 4, 4, 4],
        #   [2, 2],
        #   ... 
        # ]

        # for object detection
        if len(labels[0].shape) == 2:
            # weights for each class
            total_labels = []
            for anno in labels:
                for i in anno[:,0]:
                    total_labels.append(i)
            indices_class = list(range(len(total_labels))) # indices
        ###################################################################################

        # distribution of classes in the dataset
        df = pd.DataFrame()
        df["label"] = total_labels if len(labels[0].shape) == 2 else labels
        df.index = indices_class if len(labels[0].shape) == 2 else self.indices
        df = df.sort_index()
        label_to_count = df["label"].value_counts()

        weights = (1.0 / label_to_count[df["label"]]).tolist()

        ###################################################################################
        # for object detection
        if len(labels[0].shape) == 2:
            label_to_count /= min(label_to_count)
            # weights for each image
            weights = []
            for anno in labels:
                sum_weight = 0
                for i in anno[:,0]:
                    sum_weight += label_to_count[i]

                sum_weight /= (len(anno[:,0]) + 1) # +1 for division by zero
                weights.append(1/(sum_weight + 1)) # +1 for division by zero
        ###################################################################################

        self.weights = torch.DoubleTensor(weights)


    def _get_labels(self, dataset):
        if self.callback_get_label:
            return self.callback_get_label(dataset)
        elif isinstance(dataset, torchvision.datasets.MNIST):
            return dataset.train_labels.tolist()
        elif isinstance(dataset, torchvision.datasets.ImageFolder):
            return [x[1] for x in dataset.imgs]
        elif isinstance(dataset, torchvision.datasets.DatasetFolder):
            return dataset.samples[:][1]
        elif isinstance(dataset, torch.utils.data.Subset):
            return dataset.dataset.imgs[:][1]
        elif isinstance(dataset, torch.utils.data.Dataset):
            return dataset.labels
            # return dataset.get_labels()
        else:
            raise NotImplementedError

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples