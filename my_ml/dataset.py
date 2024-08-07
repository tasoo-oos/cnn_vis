import torch
from typing import List, Tuple
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_transforms(dataset_name: str, train: bool = True) -> transforms.Compose:
    """Get the appropriate transforms for the specified dataset."""
    if dataset_name == 'MNIST':
        transform_list = [
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize((0.1307,), (0.3081,))
        ]
        if train:
            transform_list = [
                transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=15),
            ] + transform_list + [transforms.RandomErasing(p=0.2)]
    elif dataset_name == 'FashionMNIST':
        transform_list = [
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize((0.2860,), (0.3530,)),
        ]
        if train:
            transform_list = [
                transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            ] + transform_list + [transforms.RandomErasing(p=0.1)]
    elif dataset_name == 'CIFAR10':
        transform_list = [
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ]
        if train:
            transform_list = [
                transforms.RandomResizedCrop(32, scale=(0.8, 1.0), ratio=(0.75, 1.33)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            ] + transform_list + [transforms.RandomErasing(p=0.2)]
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    return transforms.Compose(transform_list)


def load_data(dataset_name: str, batch_size: int = 64) -> Tuple[DataLoader, DataLoader]:
    """Load the specified dataset and return train and test data loaders."""
    train_transform = get_transforms(dataset_name, train=True)
    test_transform = get_transforms(dataset_name, train=False)

    if dataset_name == 'MNIST':
        train_dataset = datasets.MNIST(root='dataset/', train=True, transform=train_transform, download=True)
        test_dataset = datasets.MNIST(root='dataset/', train=False, transform=test_transform, download=True)
    elif dataset_name == 'FashionMNIST':
        train_dataset = datasets.FashionMNIST(root='dataset/', train=True, transform=train_transform, download=True)
        test_dataset = datasets.FashionMNIST(root='dataset/', train=False, transform=test_transform, download=True)
    elif dataset_name == 'CIFAR10':
        train_dataset = datasets.CIFAR10(root='dataset/', train=True, transform=train_transform, download=True)
        test_dataset = datasets.CIFAR10(root='dataset/', train=False, transform=test_transform, download=True)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader