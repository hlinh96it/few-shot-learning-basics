import os
import sys
from pathlib import Path
sys.path.append('./')

import torch
import torchvision
from torch.utils.data import DataLoader, Dataset

import pandas as pd
from PIL import Image
from config import DATA_PATH
torch.set_default_device("mps:0")


class FashionNet(Dataset):
    def __init__(self):
        super(FashionNet, self).__init__()
        self.df = pd.DataFrame(self.index_subset(DATA_PATH))
        self.df = self.df.assign(id=self.df.index.values)

        # convert string class name to int
        self.unique_characters = sorted(self.df['class_name'].unique())
        self.class_name_to_id = {self.unique_characters[i]: i for i in range(len(self.unique_characters))}
        self.df = self.df.assign(class_id=self.df['class_name'].apply(lambda c: self.class_name_to_id[c]))

        self.dataset_id_to_file_path = self.df.to_dict()['filepath']
        self.dataset_id_to_class_id = self.df.to_dict()['class_id']

        # setup transformation
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.CenterCrop(size=56),
            torchvision.transforms.Resize(size=28),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.45, 0.421], std=[0.5, 0.5, 0.5])
        ])

    def __getitem__(self, item):
        instance = Image.open(self.dataset_id_to_file_path[item]).convert('RGB')
        instance = self.transform(instance)
        return instance, self.dataset_id_to_class_id[item]

    def __len__(self):
        return len(self.df)

    @staticmethod
    def index_subset(path_to_dataset):
        """Index a subset by looping through all of its files and recording relevant information.

        :param path_to_dataset: Name of the subset
        :return: A list of dicts containing information about the image files in a particular subset
        """
        images = []
        for class_name in os.listdir(path_to_dataset):
            if not Path.is_dir(Path(path_to_dataset, class_name)): 
                continue
            
            for img in os.listdir(Path(path_to_dataset, class_name)):
                images.append({'class_name': class_name, 'filepath': Path(path_to_dataset, class_name, img)})
        return images

