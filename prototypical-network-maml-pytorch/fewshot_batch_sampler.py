import torch
from torch.utils.data import DataLoader

import random
import numpy as np
from collections import defaultdict


class FewShotBatchSampler(object):
    def __init__(self, dataset_targets: DataLoader, n_way: int, k_shot: int, include_query=False, 
                 shuffle=True, shuffle_once=False):    
        """
        Initialize the sampler. This is the method that should be called by subclasses to initialize the sampler. If you don't want to call this yourself make sure to call super (). __init__
        
        @param dataset_targets - DataLoader that contains the targets for each class
        @param n_way - Number of ways to sample ( batch size )
        @param k_shot - Number of shots to sample ( batch size )
        @param include_query - Whether to include the query targets in the batch size
        @param shuffle - Whether to shuffle the dataset before sampling. Default is True.
        @param shuffle_once - Whether to shuffle once or not. Default is
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
        """
         This function is an iterator that generates batches of indices for a few - shot learning task 
         where each batch contains samples from a few
        """
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
        """
         Number of iterations. This is the number of iterations that have been run in the run method. 
         If you want to know how many iterations have been run use : py : meth : ` iter_iter ` instead.
         @return The number of iterations in the run method as an integer. 
         Note that the number may be different from the number of iterations
        """
        return self.iterations

    def shuffle_data(self):
        """
        Randomly shuffles the data. This is useful for reproducing the data that is used to train
        """
        for c in self.classes:
            perm = torch.randperm(self.indices_per_class[c].shape[0])
            self.indices_per_class[c] = self.indices_per_class[c][perm]
        random.shuffle(self.class_list)
