import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image


class ImageDataset(Dataset):
    def __init__(self, images, targets, img_transform=None):
        super(ImageDataset, self).__init__() 
        self.images, self.targets = images, targets
        self.img_transform = img_transform
        
    def __getitem__(self, idx):
        img, target = self.images[idx], self.targets[idx]
        img = Image.fromarray(img)
        img = self.img_transform(img) if self.img_transform else img
        return img, target
    
    def __len__(self):
        return self.images.shape[0]