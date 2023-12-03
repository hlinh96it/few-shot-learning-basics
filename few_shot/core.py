import numpy as np

import torch
from torch.utils.data import Sampler, Dataset, DataLoader
from few_shot.callbacks import Callback
from few_shot.metrics import categorical_accuracy

from typing import List, Iterable, Callable, Tuple
torch.set_default_device("mps")


class NShotTaskSampler(Sampler):
    def __init__(self, dataset, episodes_per_epoch: int = 1,
                 n_shot: int = 5, k_way: int = 1, q_query: int = 1,
                 num_tasks: int = 1, fixed_tasks: List[Iterable[int]] = []):
        """Pytorch Sampler subclass that generates batches of n-shot, k-way, q-query tasks.

        :param dataset: instance to draw samples
        :param episodes_per_epoch: batches of n-shot tasks to generate in one epoch
        :param n_shot: number of examples or "shots" provided for each class during training or evaluation.
        :param k_way: number of classes involved in the task
        :param q_query: number of query examples provided for each class during evaluation
        :param num_tasks: number of n-shot tasks to group into a single batch
        :param fixed_tasks: generate tasks from the specified classes
        """
        super(NShotTaskSampler, self).__init__()
        self.episodes_per_epoch = episodes_per_epoch
        self.dataset = dataset
        assert num_tasks > 1, 'Number of tasks must be greater than 1'

        self.num_tasks = num_tasks
        self.n_shot, self.k_way, self.q_query = n_shot, k_way, q_query
        self.fixed_tasks = fixed_tasks
        self.i_task = 0

    def __len__(self):
        return self.episodes_per_epoch

    def __iter__(self):
        for _ in range(self.episodes_per_epoch):
            batch = []

            for task in range(self.num_tasks):
                if self.fixed_tasks is None:  # get random classes
                    episode_classes = np.random.choice(self.dataset.df['class_id'].unique(),
                                                       size=self.k_way, replace=False)
                else:  # loop through classes in fixed tasks
                    episode_classes = self.fixed_tasks[self.i_task % len(self.fixed_tasks)]
                    self.i_task += 1
                
                df = self.dataset.df[self.dataset.df['class_id'].isin(episode_classes)]
                
                # construct support set by random sample from class
                support_k = {k: None for k in episode_classes}
                for k in episode_classes:
                    support = df[df['class_id'] == k].sample(self.n_shot)
                    support_k[k] = support
                    for i, s in support.iterrows():
                        batch.append(s['id'])
                        
                # construct query set 
                query_k = {k: None for k in episode_classes}
                for k in episode_classes:
                    query = df[(df['class_id'] == k) & (~df['id'].isin(support_k[k]))].sample(self.q_query)
                    query_k[k] = query
                    for i, q in query.iterrows():
                        batch.append(q['id'])
                        
            yield np.stack(batch)

class EvaluateFewShot(Callback):
    def __init__(self, evaluate_func: Callable, num_tasks: int, n_shot: int, k_way: int, q_queries: int,
                 taskloader: DataLoader, prepare_batch: Callable, prefix: str='val_', **kwargs):
        """Evaluate few-shot network on an n-shot, k-way classification tasks after every epoch.

        Args:
            evaluate_func (Callable): function to perform few-shot classification, ie prototype_net
            num_tasks (int): number of n-shot classification tasks
            n_shot (int): number of sample for each class 
            k_way (int): number of classes in n-shot 
            q_queries (int): number of samples of query
            taskloader (DataLoader): instance of NShotWrapper class
            prepare_batch (Callable): pre-processing function to apply to samples from dataset
            prefix (str, optional): prefix to identify dataset. Defaults to 'val_'.
        """
        super(EvaluateFewShot, self).__init__()
        self.evaluate_func, self.num_tasks = evaluate_func, num_tasks
        self.n_shot, self.k_way, self.q_queries = n_shot, k_way, q_queries
        self.task_loader = taskloader
        self.prepare_batch = prepare_batch
        self.prefix = prefix
        self.kwargs = kwargs
        self.metric_name = f'{self.prefix}{self.n_shot}-shot_{self.k_way}-way-acc'
        
    def on_train_begin(self, logs=None):
        self.loss_function = self.params['loss_fn']
        self.optimizer = self.params['optimizer']
        
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        seen = 0
        totals = {'loss': 0, self.metric_name: 0}
        for batch_index, batch in enumerate(self.task_loader):
            x, y = self.prepare_batch(batch)
            loss, y_pred = self.evaluate_func(
                self.model, self.optimizer, self.loss_function, x, y,
                n_shot=self.n_shot, k_way=self.k_way, q_queries=self.q_queries,
                train=False, **self.kwargs
            )
            seen += y_pred.shape[0]  # used to calculate mean loss
            totals['loss'] += loss.item() * y_pred.shape[0]
            totals[self.metric_name] += categorical_accuracy(y, y_pred) * y_pred.shape[0]
            
        logs[self.prefix + 'loss'] = totals['loss'] / seen
        logs[self.metric_name] = totals[self.metric_name] / seen

def create_nshot_task_label(k: int, q: int) -> torch.Tensor:
    """Creates an n-shot task label.
    Label has the structure:
        [0]*q + [1]*q + ... + [k-1]*q
    # Arguments
        k: Number of classes in the n-shot classification task
        q: Number of query samples for each class in the n-shot classification task
    # Returns
        y: Label vector for n-shot task of shape [q * k, ]
    """
    y = torch.arange(0, k, 1 / q).long()
    return y