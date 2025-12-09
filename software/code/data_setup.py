import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

num_workers = os.cpu_count()

def create_dataloader(directory: str,
                      transform: transforms.Compose,
                      batch_size: int,
                      shuffle: bool):
    
    data = datasets.ImageFolder(directory, transform=transform)

    dataloader = DataLoader(
        data,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )

    return dataloader

def get_class_names(directory: str):
    data = datasets.ImageFolder(directory)
    return data.classes