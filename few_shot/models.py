import sys
sys.path.append('./')
import config

import torch
import torch.nn as nn
import torch.functional as F
import torchvision

from typing import Dict
torch.set_default_device("mps:0")


class FewShotClassifer(nn.Module):
    def __init__(self, num_input_channels: int, k_way: int, final_layer_size: int=64):
        """Creates a few-shot classifer as used in MAML paper.

        Args:
            num_input_channels (int): number of color channels the model expects the input data to contain
            k_way (int): number of classes to discriminate
            final_layer_size (int, optional): total output of data. Defaults to 64.
        """
        super(FewShotClassifer, self).__init__()
        self.conv1 = conv_block(num_input_channels, 64)
        self.conv2 = conv_block(in_channels=64, out_channels=64)
        self.conv3 = conv_block(in_channels=64, out_channels=64)
        self.conv4 = conv_block(in_channels=64, out_channels=64)
        self.logits = nn.Linear(in_features=final_layer_size, out_features=k_way)
        
    def forward(self, x):
        x = self.conv4(self.conv3(self.conv2(self.conv1(x.to(config.DEVICE)))))
        x = x.view(x.size(0), -1)
        return self.logits(x)
        
        
def conv_block(in_channels: int, out_channels: int) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )
        
    
        
        