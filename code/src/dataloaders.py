import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms, datasets

name2dataset = {
    'MNIST': datasets.MNIST,
    'FashionMNIST': datasets.FashionMNIST,
    'CIFAR10': datasets.CIFAR10,
    'CIFAR100': datasets.CIFAR100,
}

def init_dataloaders(dataset_name, 
                     batch_size = 64,
                     path_to_dataset = None):
    if 'CIFAR' not in dataset_name:
        transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])
    else:
        transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.4915, 0.4823, .4468), (0.2470, 0.2435, 0.2616))])

    if path_to_dataset is None:
        path_to_dataset = f'./loaded_datasets/'
    train_dataset = name2dataset[dataset_name](path_to_dataset, download=True, train=True, transform=transform)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    return train_dataloader