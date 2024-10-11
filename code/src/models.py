import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torchvision import datasets, transforms

from omegaconf import OmegaConf

class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out, kernel_size):
        super(ConvBlock, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=kernel_size, stride=1, padding=0),
            nn.ReLU()
        )
    def forward(self, x):
        return self.body(x)

        
class ConvNet(nn.Module):
    def __init__(self, channels_list, ker_size_list, classes, input_sizes):
        super(ConvNet, self).__init__()
        self.model = self._stack_conv_blocks(channels_list, ker_size_list)
        self.flatten = nn.Flatten()

        flat_size = self._calc_flattened_size(channels_list, ker_size_list, input_sizes)
        self.head = nn.Sequential(
            nn.Linear(flat_size, classes)
        )

    def _calc_flattened_size(self, channels_list, ker_size_list, input_sizes):
        ker_delta = np.sum(ker_size_list) - len(ker_size_list)
        final_sizes = (input_sizes[0] - ker_delta, input_sizes[1] - ker_delta)
        return final_sizes[0]*final_sizes[1]*channels_list[-1]

    def _stack_conv_blocks(self, channels_list, ker_size_list):
        convs_list = []
        for c_in, c_out, kernel_size in zip(channels_list[:-2], channels_list[1:-1], ker_size_list[:-1]):
            convs_list.append(ConvBlock(c_in, c_out, kernel_size))
        convs_list.append(nn.Conv2d(channels_list[-2], channels_list[-1], ker_size_list[-1]))
        return nn.Sequential(*convs_list)

    def forward(self, images):
        out = self.model(images)
        flattened = self.flatten(out)
        logits = self.head(flattened)
        return logits

def make_default_model_use_layers_num(config, layers_num):
    channels_list = [config.input_channels]+[config.channels]*layers_num
    ker_size_list = [config.kernel_size]*layers_num
    return ConvNet(channels_list, ker_size_list, config.classes, tuple(config.input_size))