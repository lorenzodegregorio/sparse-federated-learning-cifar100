import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, random_split
import numpy as np
import random
from torchvision import transforms

def load_cifar100(validation_split=0.1):
    transform = transforms.Compose([
        transforms.Resize(224),  # required for ViT
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409),
                             (0.2673, 0.2564, 0.2761))
    ])

    full_trainset = torchvision.datasets.CIFAR100(
        root='./data', train=True, download=True, transform=transform)

    testset = torchvision.datasets.CIFAR100(
        root='./data', train=False, download=True, transform=transform)

    total_train = len(full_trainset)
    val_size = int(validation_split * total_train)
    train_size = total_train - val_size

    train_subset, val_subset = random_split(full_trainset, [train_size, val_size])

    return train_subset, val_subset, testset


def split_data_iid(dataset, num_clients=100):
    """
    Splits dataset IID among clients.
    Each client receives an (approximately) equal number of random samples.
    """
    indices = list(range(len(dataset)))
    random.shuffle(indices)

    # Split indices among clients
    client_indices = np.array_split(indices, num_clients)
    client_data = {i: Subset(dataset, client_indices[i]) for i in range(num_clients)}

    return client_data

import numpy as np
from torch.utils.data import Subset

def split_data_noniid(dataset, num_clients, num_classes_per_client, seed=42):
    """
    Splits a dataset into non-IID partitions based on class labels.
    Each client receives data from `num_classes_per_client` random classes.
    
    Supports both standard datasets and torch.utils.data.Subset.
    """
    np.random.seed(seed)

    # Handle Subset (e.g., from train-validation split)
    if isinstance(dataset, Subset):
        full_targets = dataset.dataset.targets
        subset_indices = dataset.indices
        targets = np.array([full_targets[i] for i in subset_indices])
        index_pool = np.array(subset_indices)
        original_to_subset_map = {idx: i for i, idx in enumerate(subset_indices)}
    else:
        # Full dataset
        targets = np.array(dataset.targets)
        index_pool = np.arange(len(targets))
        original_to_subset_map = None

    num_classes = len(np.unique(targets))
    if num_classes_per_client > num_classes:
        raise ValueError(f"num_classes_per_client ({num_classes_per_client}) cannot be greater than total number of classes ({num_classes})")

    # Collect indices for each class
    class_indices = {i: index_pool[targets == i] for i in range(num_classes)}
    for i in class_indices:
        np.random.shuffle(class_indices[i])

    # Assign classes to each client
    client_classes = [np.random.choice(num_classes, num_classes_per_client, replace=False)
                      for _ in range(num_clients)]

    # Count how many clients want each class
    clients_per_class = {i: sum(i in cc for cc in client_classes) for i in range(num_classes)}

    # How many samples each client should get per class
    samples_per_class_client = {
        i: len(class_indices[i]) // max(1, clients_per_class[i])
        for i in range(num_classes)
    }

    class_offset = {i: 0 for i in range(num_classes)}
    client_idx_dict = {}

    for client_id, client_class_list in enumerate(client_classes):
        client_indices = []

        for class_id in client_class_list:
            start = class_offset[class_id]
            count = samples_per_class_client[class_id]
            end = start + count

            original_indices = class_indices[class_id][start:end]
            class_offset[class_id] = end

            # Map original indices to subset indices if needed
            if original_to_subset_map:
                subset_indices = [original_to_subset_map[idx] for idx in original_indices if idx in original_to_subset_map]
                client_indices.extend(subset_indices)
            else:
                client_indices.extend(original_indices.tolist())

        np.random.shuffle(client_indices)
        client_idx_dict[client_id] = client_indices

    return client_idx_dict
