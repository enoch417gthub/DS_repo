"""
config.py
Single source of truth for all hyperparameters.
"""

import torch
import os

CONFIG = {
    # Data
    "data_dir": "data",
    "img_size": 224,
    "num_classes": 4,
    "class_names": ["eosinophil", "lymphocyte", "monocyte", "neutrophil"],
    
    # Training
    "batch_size": 32,
    "num_epochs": 60,
    "num_workers": 4,
    "pin_memory": True,
    
    # Optimiser
    "optimizer": "adamw",  # "adam" | "adamw" | "sgd"
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "momentum": 0.9,  # for SGD only
    
    # Scheduler
    "scheduler": "cosine",  # "cosine" | "step" | "plateau"
    "warmup_epochs": 5,
    "min_lr_ratio": 0.01,
    
    # Regularisation
    "dropout_rate": 0.4,
    "label_smoothing": 0.1,
    "patience": 12,  # early stopping
    
    # Architecture
    "base_filters": 32,
    "num_blocks": 5,
    
    # System
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "seed": 42,
    "checkpoint_dir": "checkpoints",
    "log_dir": "runs",
}

# Dataset statistics (compute these from your training set)
# For now, using ImageNet stats. Replace with your computed stats after EDA.
MEAN = [0.6787297129631042, 0.6414586305618286, 0.6606341004371643]
STD = [0.2623903453350067, 0.2615512013435364, 0.25914466381073]

def set_seed(seed):
    """Set all random seeds for reproducibility."""
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False