"""
scheduler.py
Learning rate schedulers, including warmup + cosine decay.
"""

import math
from torch.optim.lr_scheduler import LambdaLR, _LRScheduler


class WarmupCosineScheduler(_LRScheduler):
    """
    Warmup + Cosine Annealing scheduler.
    
    Linearly increases lr from 0 to base_lr over warmup_epochs,
    then decays following cosine curve to min_lr over remaining epochs.
    
    This is best practice for training CNNs from scratch.
    """
    
    def __init__(self, optimizer, warmup_epochs, total_epochs, 
                 min_lr_ratio=0.01, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr_ratio = min_lr_ratio
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Warmup phase: linear increase
            alpha = self.last_epoch / self.warmup_epochs
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            # Cosine decay phase
            progress = (self.last_epoch - self.warmup_epochs) / \
                      (self.total_epochs - self.warmup_epochs)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            # Scale from min_lr_ratio up to 1, then apply cosine
            scale = self.min_lr_ratio + (1 - self.min_lr_ratio) * cosine_decay
            return [base_lr * scale for base_lr in self.base_lrs]


def get_scheduler(optimizer, config):
    """
    Factory function to get the appropriate scheduler.
    
    Args:
        optimizer: PyTorch optimizer
        config: Configuration dictionary
    
    Returns:
        Learning rate scheduler
    """
    scheduler_type = config["scheduler"]
    
    if scheduler_type == "cosine":
        return WarmupCosineScheduler(
            optimizer,
            warmup_epochs=config["warmup_epochs"],
            total_epochs=config["num_epochs"],
            min_lr_ratio=config["min_lr_ratio"],
        )
    elif scheduler_type == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=20, gamma=0.5
        )
    elif scheduler_type == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=5, factor=0.5
        )
    else:
        # Constant LR (default)
        return None