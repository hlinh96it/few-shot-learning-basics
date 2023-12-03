import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config
from few_shot.datasets import FashionNet
from few_shot.core import NShotTaskSampler, EvaluateFewShot
from few_shot.models import FewShotClassifer
from torchsummary import summary
torch.set_default_device("mps")


# %% Data preprocessing
dataset = config.DATASET
n, k, q = config.n, config.k, config.q
inner_train_steps, inner_val_steps = config.inner_train_steps, config.inner_val_steps
inner_learning_rate, meta_learning_rate = config.inner_train_steps, config.meta_learning_rate
meta_batch_size, order = config.meta_batch_size, config.order
epochs, epoch_len, eval_batches = config.epochs, config.epoch_len, config.eval_batches

if dataset == 'fashionNet':
    n_epochs, dataset_class = 50, FashionNet()
    num_input_channels = 3
    drop_learning_every = 30
else:
    raise ValueError

print(f'Dataset: {dataset} | Order: {order} | n, k, q: {n, k, q}')

# %% Create dataset
dataset = DataLoader(dataset_class, num_workers=8,
                     batch_sampler=NShotTaskSampler(dataset=dataset_class, episodes_per_epoch=epoch_len,
                                                             n_shot=n, k_way=k, q_query=q,
                                                             num_tasks=num_input_channels))

# %% Prepare model and train
meta_model = FewShotClassifer(num_input_channels, k_way=k, final_layer_size=64)
meta_optimizer = torch.optim.Adam(params=meta_model.parameters(), lr=config.meta_learning_rate)
loss_function = nn.CrossEntropyLoss()

# %% Fit model

