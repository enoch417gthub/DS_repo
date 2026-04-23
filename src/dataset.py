"""
dataset.py
Data loading, transforms, and augmentation pipeline.
"""

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
import os
from config import CONFIG, MEAN, STD


def get_transforms(img_size=224, mode="train"):
    """
    Get transformation pipeline for images.
    
    Args:
        img_size: Target image size
        mode: "train" (with augmentation) or "val"/"test" (without)
    
    Returns:
        torchvision.transforms.Compose pipeline
    """
    normalize = transforms.Normalize(mean=MEAN, std=STD)
    
    if mode == "train":
        return transforms.Compose([
            # Slight resize then random crop for scale invariance
            transforms.Resize((img_size + 20, img_size + 20)),
            transforms.RandomCrop(img_size),
            
            # Geometric augmentations (valid for blood cells)
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=25),
            
            # Colour augmentations (handles staining variation)
            transforms.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.2,
                hue=0.05
            ),
            
            # Simulates focus variation
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
            
            # Convert to tensor and normalise
            transforms.ToTensor(),
            normalize,
            
            # Simulates occlusion by other cells/debris
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
        ])
    else:
        # Validation/Test: no augmentation, just resize and normalise
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            normalize,
        ])


def get_dataloaders(config):
    """
    Create train, validation, and test dataloaders.
    
    Expects folder structure:
    data/
        train/
            eosinophil/
            lymphocyte/
            monocyte/
            neutrophil/
        val/
            same subfolders
        test/
            same subfolders
    
    Args:
        config: Configuration dictionary
    
    Returns:
        dict: {"train": loader, "val": loader, "test": loader}
        list: class_names
    """
    data_dir = config["data_dir"]
    batch_size = config["batch_size"]
    num_workers = config["num_workers"]
    
    # Create datasets
    train_dataset = datasets.ImageFolder(
        os.path.join(data_dir, "train"),
        transform=get_transforms(config["img_size"], mode="train")
    )
    
    val_dataset = datasets.ImageFolder(
        os.path.join(data_dir, "val"),
        transform=get_transforms(config["img_size"], mode="val")
    )
    
    test_dataset = datasets.ImageFolder(
        os.path.join(data_dir, "test"),
        transform=get_transforms(config["img_size"], mode="val")
    )
    
    # Handle class imbalance with WeightedRandomSampler
    class_counts = []
    for class_idx in range(len(train_dataset.classes)):
        class_counts.append(len([y for _, y in train_dataset.samples if y == class_idx]))
    
    class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
    sample_weights = [class_weights[label] for _, label in train_dataset.samples]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,  # Use sampler instead of shuffle
        num_workers=num_workers,
        pin_memory=config["pin_memory"],
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=config["pin_memory"],
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=config["pin_memory"],
    )
    
    loaders = {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
    }
    
    return loaders, train_dataset.classes


def compute_dataset_stats(data_dir, img_size=224):
    """
    Compute mean and std of your dataset (run once, then update config).
    
    Args:
        data_dir: Path to training images folder
        img_size: Target image size
    
    Returns:
        mean, std: Lists of 3 values each
    """
    raw_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    
    dataset = datasets.ImageFolder(
        os.path.join(data_dir, "train"),
        transform=raw_transform
    )
    
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)
    
    mean = torch.zeros(3)
    std = torch.zeros(3)
    n_samples = 0
    
    for imgs, _ in loader:
        batch_samples = imgs.size(0)
        # mean over H,W for each channel
        mean += imgs.mean(dim=[0, 2, 3]) * batch_samples
        std += imgs.std(dim=[0, 2, 3]) * batch_samples
        n_samples += batch_samples
    
    mean /= n_samples
    std /= n_samples
    
    print(f"Dataset mean: {mean.tolist()}")
    print(f"Dataset std: {std.tolist()}")
    
    return mean.tolist(), std.tolist()