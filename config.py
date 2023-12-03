import os
import torch

PATH = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = '/Users/hoanglinh96nl/FashionNet Dataset/myntradataset'
EPSILON = 1e-8
DEVICE = torch.device('mps')

DATASET = 'fashionNet'
n = 1
k = 5
q = 1
inner_train_steps = 1
inner_val_steps = 1
inner_learning_rate = 1
meta_learning_rate = 0.4
meta_batch_size = 32
order = 1
epochs = 20
epoch_len = 100
eval_batches = 20

