import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from omegaconf import OmegaConf

from src.dataloaders import init_dataloaders
import json

def train(model,
          optimizer,
          criterion,
          train_dataloader,
          valid_dataloader,
          epochs,
          device):

    losses = []
    model.train()

    all_ans = 0
    right_ans = 0

    for epoch in range(epochs):
        for images, labels in train_dataloader:
            images, labels = images.to(device).to(torch.float), labels.to(torch.long).to(device)
    
            logits = model(images)
            prediction = torch.argmax(logits, dim = 1)

            if epoch == epochs - 1:
                right_ans += torch.sum(prediction == labels).item()
                all_ans += len(images)
    
            loss = criterion(logits, labels)
            optimizer.zero_grad()
    
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
    print(right_ans/all_ans)
    print()
    return losses

def valid(model,
          criterion,
          valid_dataloader,
          device):
    
    model.eval()
    losses = []
    
    right_ans = 0
    all_ans = 0
    
    for _ in range(1):
        for images, labels in valid_dataloader:
            images, labels = images.to(device).to(torch.float), labels.to(torch.long).to(device)
    
            logits = model(images) # bs, C
            prediction = torch.argmax(logits, dim = 1)
            right_ans += torch.sum(prediction == labels).item()
            all_ans += len(labels)

            for i in range(len(images)):
                loss = criterion(logits[i], labels[i])
                losses.append(loss.item())
    print(right_ans/all_ans)
    return losses
    
def train_models_with_changed_params(config, model_create_func, device = None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_loader = init_dataloaders(config.dataset_name, 
                                    batch_size=config.train_batch_size)
    for param in range(*(config.layers_num_range)):
        criterion = nn.CrossEntropyLoss()
        model = model_create_func(config, param).to(device)
        opt = optim.Adam(model.parameters(), lr = config.learning_rate)
        
        train(model, opt, criterion, train_loader, train_loader, config.num_epochs, device)

        path = config.save_model_path_prefix + '_' + str(param) + '.pt'
        torch.save(model.state_dict(), path)

def extract_models_results_with_changed_params(config, 
                                               model_create_func, 
                                               device = None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_loader = init_dataloaders(config.dataset_name, 
                                    batch_size=config.train_batch_size)
    
    results = [] 
    
    for param in range(*(config.layers_num_range)):
        criterion = nn.CrossEntropyLoss()
        model = model_create_func(config, param).to(device)
        path = config.save_model_path_prefix + '_' + str(param) + '.pt'
        model.load_state_dict(torch.load(path, 
                                         weights_only = True))
        losses = valid(model, criterion, train_loader, device)
        results.append({
            'param':param,
            'losses':losses
        })
    with open(config.save_results_path, 'w') as f:
        json.dump(results, f, indent = 4)
