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

class ConvNetWithPool(nn.Module):
    def __init__(self, channels_list, ker_size_list, 
                 classes, input_sizes, 
                 pool_position, pool_type):
        super(ConvNetWithPool, self).__init__()
        self.flat_size = self._calc_flattended_size(channels_list, ker_size_list, input_sizes, pool_position)
        self._build_conv_laysers(channels_list, ker_size_list, pool_position, pool_type)
        self.flat = nn.Flatten()
        self.fc = nn.Linear(self.flat_size, classes)

    def _build_conv_laysers(self, 
                            channels_list, ker_size_list, 
                            pool_position, pool_type):
        self.conv_part = []
        for i, (ker, in_channels, out_channels) in enumerate(zip(ker_size_list, channels_list[:-1], channels_list[1:])):
            self.conv_part.append(nn.Conv2d(in_channels, out_channels, ker))
            if i != pool_position:
                if i != len(ker_size_list):
                    self.conv_part.append(nn.ReLU())
            elif pool_type == 'max':
                self.conv_part.append(nn.MaxPool2d(2))
            elif pool_type == 'avg':
                self.conv_part.append(nn.AvgPool2d(2))
        self.conv_part = nn.Sequential(*self.conv_part) 

    def _calc_flattended_size(self, 
                              channels_list, ker_size_list, 
                              input_sizes, 
                              pool_position):
        size = input_sizes[0]
        for i, ker in enumerate(ker_size_list):
            size = size - ker + 1
            if i == pool_position:
                size = size // 2
        size = size**2
        size = size*channels_list[-1]
        return size
        
    def forward(self, input):
        return self.fc(self.flat(self.conv_part(input)))


def make_default_model_use_layers_num(config, layers_num):
    channels_list = [config.input_channels]+[config.channels]*layers_num
    ker_size_list = [config.kernel_size]*layers_num
    return ConvNet(channels_list, ker_size_list, config.classes, tuple(config.input_size))

def make_default_model_use_kernel_size(config, kernel_size):
    layers_num = config.layers_num
    channels_list = [config.input_channels]+[config.channels]*layers_num
    ker_size_list = [kernel_size]*layers_num
    return ConvNet(channels_list, ker_size_list, config.classes, tuple(config.input_size))

def make_default_model_use_channel_size(config, channels):
    channels_list = [config.input_channels] + [channels]*config.layers_num
    ker_size_list = [config.kernel_size]*config.layers_num
    return ConvNet(channels_list, ker_size_list, config.classes, tuple(config.input_size))

def make_default_model_use_maxpool_position(config, pos):
    channels_list = [config.input_channels] + [config.channels]*config.layers_num
    ker_size_list = [config.kernel_size]*config.layers_num
    return ConvNetWithPool(channels_list, ker_size_list, config.classes, tuple(config.input_size), pos, 'max')

def make_default_model_use_avgpool_position(config, pos):
    channels_list = [config.input_channels] + [config.channels]*config.layers_num
    ker_size_list = [config.kernel_size]*config.layers_num
    return ConvNetWithPool(channels_list, ker_size_list, config.classes, tuple(config.input_size), pos, 'avg')
