import torch
import os
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import copy
import torch

def evaluate_model(model, dataloader, device):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

    accuracy = correct / total
    return accuracy

def federated_aggregation(client_models, client_weights, global_model):
    new_state_dict = copy.deepcopy(global_model.state_dict())
    total_samples = sum(client_weights)

    for key in new_state_dict.keys():
        weighted_sum = 0
        for state_dict, weight in zip(client_models, client_weights):
            weighted_sum += state_dict[key] * (weight / total_samples)
        new_state_dict[key] = weighted_sum

    global_model.load_state_dict(new_state_dict)
    return global_model