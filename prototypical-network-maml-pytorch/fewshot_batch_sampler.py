import torch
from torch.utils.data import DataLoader

import random
import numpy as np
from collections import defaultdict


class FewShotBatchSampler(object):
    def __init__(self, dataset_targets: DataLoader, n_way: int, k_shot: int, include_query=False, 
                 shuffle=True, shuffle_once=False):    
        """
        Initializes the FewShotBatchSampler
        Args:
            dataset_targets: DataLoader: The dataloader containing the dataset
            n_way: int: Number of classes per batch
            k_shot: int: Number of examples per class
            include_query: bool: Whether to include query samples
            shuffle: bool: Whether to shuffle the data
            shuffle_once: bool: Whether to shuffle only once
        """
        super(FewShotBatchSampler, self).__init__()

        self.dataset_targets = dataset_targets
        self.n_way, self.k_shot, self.include_query = n_way, k_shot, include_query
        self.shuffle = shuffle

        # if include_query is true then the k_shot is multiplied by k_shot
        if self.include_query:
            self.k_shot *= 2  # if use query, batch size extend to query size
        self.batch_size = self.n_way * self.k_shot

        # organize samples by classes
        self.classes = torch.unique(self.dataset_targets).tolist()
        self.num_classes = len(self.classes)
        self.indices_per_class = {}
        self.batch_per_class = {}
        
        # This method is used to calculate the indices of the classes in the dataset.
        for c in self.classes:
            self.indices_per_class[c] = torch.where(self.dataset_targets == c)[0]
            self.batch_per_class[c] = self.indices_per_class[c].shape[0] // self.k_shot

        # create a list of classes from which we select the N classes per batch
        self.iterations = sum(self.batch_per_class.values()) // self.n_way
        self.class_list = [c for c in self.classes for _ in range(self.batch_per_class[c])]
        
        # Reorder the data in the dataset.
        if shuffle_once or self.shuffle:
            self.shuffle_data()
        else:
            sort_idx = [i + p*self.num_classes for i, c in enumerate(self.classes) for p in range(self.batch_per_class[c])]
            self.class_list = np.array(self.class_list)[np.argsort(sort_idx)].tolist()

    def __iter__(self):
        # Shuffle the data in the data set.
        if self.shuffle:
            self.shuffle_data()

        # in case no key in the dict, used to store index of sample to add to batch
        start_index = defaultdict(int)
        # generator for each iteration of the iteration
        for it in range(self.iterations):
            # select N classes for a batch
            class_batch = self.class_list[it*self.n_way: (it+1)*self.n_way]
            index_batch = []
            # This method is used to select k samples for each class
            for c in class_batch:
                # for each class, we select k samples and add to a batch
                index_batch.extend(
                    self.indices_per_class[c][start_index[c]: start_index[c] + self.k_shot])
                start_index[c] += self.k_shot
            # Include the query in the index_batch.
            if self.include_query:
                index_batch = index_batch[::2] + index_batch[1::2]
            yield index_batch

    def __len__(self):
        return self.iterations

    def shuffle_data(self):
        for c in self.classes:
            perm = torch.randperm(self.indices_per_class[c].shape[0])
            self.indices_per_class[c] = self.indices_per_class[c][perm]
        random.shuffle(self.class_list)
