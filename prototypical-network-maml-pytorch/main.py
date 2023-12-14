import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as torch_transform
from torchvision.datasets import CIFAR100

from dataloader import ImageDataset
from fewshot_batch_sampler import FewShotBatchSampler
from trainer import train_model, test_model
from prototypical_net import PrototypicalNet
device = torch.device('mps')

def dataset_from_labels(imgs, targets, class_set, **kwargs):
    class_mask = (targets[:, None] == class_set[None, :]).any(dim=-1)
    return ImageDataset(images=imgs[class_mask], targets=targets[class_mask], **kwargs)


if __name__ == '__main__':
    DATA_MEANS = torch.Tensor([0.5183975 , 0.49192241, 0.44651328])
    DATA_STD = torch.Tensor([0.26770132, 0.25828985, 0.27961241])
    n_way = 5
    k_shot = 32

    # %% Load CIFAR100 dataset
    cifar_train_set = CIFAR100(root='/Users/hoanglinh96nl', download=True, train=True,
                            transform=torchvision.transforms.ToTensor())
    cifar_test_set = CIFAR100(root='/Users/hoanglinh96nl', download=True, train=False,
                            transform=torchvision.transforms.ToTensor())

    cifar_all_images = np.concatenate([cifar_train_set.data, cifar_test_set.data], axis=0)
    cifar_all_labels = torch.LongTensor(cifar_train_set.targets + cifar_test_set.targets)

    # %% Split data to train and test, original we have 100 classes
    classes = torch.randperm(100)  # random 100 numbers with range of 0 to 99 ~ shuffle
    train_classes, val_classes, test_classes = classes[:80], classes[80: 90], classes[90: ]

    # %% DataLoader and transformation
    train_transforms = torch_transform.Compose([torch_transform.RandomHorizontalFlip(),
                                            torch_transform.RandomResizedCrop(
                                                size=(32, 32), scale=(0.8, 1.0), ratio=(0.9, 1.1)),
                                                torch_transform.ToTensor(), 
                                                torch_transform.Normalize(DATA_MEANS, DATA_STD)])
    test_transforms = torch_transform.Compose(
        [torch_transform.ToTensor(), torch_transform.Normalize(DATA_MEANS, DATA_STD)])

    train_set = dataset_from_labels(cifar_all_images, cifar_all_labels, train_classes, img_transform=train_transforms)
    validation_set = dataset_from_labels(cifar_all_images, cifar_all_labels, val_classes, img_transform=test_transforms)
    test_set = dataset_from_labels(cifar_all_images, cifar_all_labels, test_classes, img_transform=test_transforms)

    train_dataloader = DataLoader(dataset=train_set, num_workers=9, persistent_workers=True, 
                                batch_sampler=FewShotBatchSampler(dataset_targets=train_set.targets, n_way=n_way,
                                                                    k_shot=k_shot, include_query=True, shuffle=True))
    val_dataloader = DataLoader(dataset=validation_set, num_workers=9, persistent_workers=True,
                                batch_sampler=FewShotBatchSampler(dataset_targets=validation_set.targets, n_way=n_way,
                                                                k_shot=k_shot, include_query=True, shuffle=False, shuffle_once=True))

    # %% Training few-shot learning model
    model = train_model(model_class=PrototypicalNet, proto_dim=128, learning_rate=0.001,
                        train_loader=train_dataloader, val_loader=val_dataloader)
    
    # %% Test model
    test_dataloader = DataLoader(test_set, num_workers=9, persistent_workers=True,
                                 batch_sampler=FewShotBatchSampler(test_set.targets, n_way, k_shot))
    test_model(PrototypicalNet, test_dataloader)