import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_transforms(train: bool = True):
    """
    Returns image transformations aligned with pretrained CNN requirements.
    """
    if train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],   # ImageNet mean
                std=[0.229, 0.224, 0.225]     # ImageNet std
            )
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])


def get_dataloaders(batch_size: int = 32, num_workers: int = 2):
    """
    Returns training and validation dataloaders for Fashion-MNIST.
    """

    train_dataset = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=get_transforms(train=True)
    )

    val_dataset = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=get_transforms(train=False)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    return train_loader, val_loader


# How this prepares for baseline training
# Data standardized for Baseline CNN
# Correct input shape for Pretrained CNN
# Clean seperation of concerns for MLflow
# Same preprocessing can be reused for FastAPI
