"""
train.py
Complete training loop with train_one_epoch, validate, early stopping.
"""

import torch
import os
import numpy as np
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

from config import CONFIG, set_seed
from model import BloodCellCNN
from dataset import get_dataloaders
from scheduler import get_scheduler


class EarlyStopping:
    """
    Monitors val loss. Saves model when a new best is found.
    Stops training if no improvement for 'patience' epochs.
    """
    def __init__(self, patience=12, min_delta=1e-4, path="checkpoints/best.pth"):
        self.patience = patience
        self.min_delta = min_delta
        self.path = path
        self.best_loss = float("inf")
        self.counter = 0
        self.best_epoch = 0
    
    def __call__(self, val_loss, model, epoch):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_epoch = epoch
            torch.save(model.state_dict(), self.path)
            return False  # Don't stop
        else:
            self_counter += 1
            if self.counter >= self.patience:
                print(f"\nEarly stop at epoch {epoch}. Best: epoch {self.best_epoch}")
                return True  # Stop training
            return False


def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None):
    """
    Run one full pass over the training set.
    
    Returns:
        avg_loss, accuracy
    """
    model.train()  # CRITICAL: enables dropout and batchnorm train mode
    
    total_loss = 0.0
    correct = 0
    n_samples = 0
    
    pbar = tqdm(loader, desc="Train", leave=False)
    for imgs, labels in pbar:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        # Step 1: Clear old gradients
        optimizer.zero_grad()
        
        # Step 2: Forward pass (with mixed precision if scaler exists)
        if scaler is not None:
            with autocast():
                logits = model(imgs)
                loss = criterion(logits, labels)
        else:
            logits = model(imgs)
            loss = criterion(logits, labels)
        
        # Step 3: Backward pass
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            # Gradient clipping: prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        # Accumulate metrics
        total_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        n_samples += imgs.size(0)
        
        # Update progress bar
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    avg_loss = total_loss / n_samples
    accuracy = correct / n_samples
    
    return avg_loss, accuracy


def validate(model, loader, criterion, device):
    """
    Run one full pass over the validation set.
    NO gradient computation, NO weight updates.
    
    Returns:
        avg_loss, accuracy
    """
    model.eval()  # CRITICAL: disables dropout, switches BN to eval mode
    
    total_loss = 0.0
    correct = 0
    n_samples = 0
    
    with torch.no_grad():  # no gradient tracking - saves memory and time
        pbar = tqdm(loader, desc="Val", leave=False)
        for imgs, labels in pbar:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            logits = model(imgs)
            loss = criterion(logits, labels)
            
            total_loss += loss.item() * imgs.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            n_samples += imgs.size(0)
    
    avg_loss = total_loss / n_samples
    accuracy = correct / n_samples
    
    return avg_loss, accuracy


def train(config):
    """
    Main training function.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        history: dict of training metrics
        class_names: list of class names
    """
    # Set seed for reproducibility
    set_seed(config["seed"])
    
    # Create directories
    os.makedirs(config["checkpoint_dir"], exist_ok=True)
    os.makedirs(config["log_dir"], exist_ok=True)
    
    device = config["device"]
    print(f"Using device: {device}")
    
    # Load data
    print("\nLoading data...")
    loaders, class_names = get_dataloaders(config)
    
    # Print dataset sizes
    print(f"Train samples: {len(loaders['train'].dataset)}")
    print(f"Val samples: {len(loaders['val'].dataset)}")
    print(f"Test samples: {len(loaders['test'].dataset)}")
    print(f"Classes: {class_names}")
    
    # Create model
    print("\nCreating model...")
    model = BloodCellCNN(
        num_classes=config["num_classes"],
        base_filters=config["base_filters"],
        dropout_rate=config["dropout_rate"],
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss function with label smoothing
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=config["label_smoothing"])
    
    # Optimiser
    opt_name = config["optimizer"]
    if opt_name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config["lr"],
            weight_decay=config["weight_decay"],
        )
    elif opt_name == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config["lr"],
        )
    elif opt_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config["lr"],
            momentum=config["momentum"],
            weight_decay=config["weight_decay"],
            nesterov=True,
        )
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")
    
    # Scheduler
    scheduler = get_scheduler(optimizer, config)
    
    # Mixed precision scaler (for GPU)
    scaler = GradScaler() if device == "cuda" else None
    
    # TensorBoard writer
    writer = SummaryWriter(config["log_dir"])
    
    # Early stopping
    stopper = EarlyStopping(
        patience=config["patience"],
        path=f"{config['checkpoint_dir']}/best.pth"
    )
    
    # Training history
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "lr": [],
    }
    
    print(f"\n{'='*70}")
    print(f"Training Configuration:")
    print(f"  Epochs: {config['num_epochs']}")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Learning rate: {config['lr']}")
    print(f"  Optimizer: {opt_name}")
    print(f"  Scheduler: {config['scheduler']}")
    print(f"{'='*70}\n")
    
    # Training loop
    for epoch in range(1, config["num_epochs"] + 1):
        print(f"\nEpoch {epoch}/{config['num_epochs']}")
        print("-" * 40)
        
        # Train
        train_loss, train_acc = train_one_epoch(
            model, loaders["train"], optimizer, criterion, device, scaler
        )
        
        # Validate
        val_loss, val_acc = validate(
            model, loaders["val"], criterion, device
        )
        
        # Update scheduler (different for plateau)
        if config["scheduler"] == "plateau":
            scheduler.step(val_loss)
        elif scheduler is not None:
            scheduler.step()
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]["lr"]
        
        # Record history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)
        
        # Log to TensorBoard
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Accuracy/train", train_acc, epoch)
        writer.add_scalar("Accuracy/val", val_acc, epoch)
        writer.add_scalar("LR", current_lr, epoch)
        
        # Print metrics
        print(f"Train: loss={train_loss:.4f}, acc={train_acc:.4f}")
        print(f"Val:   loss={val_loss:.4f}, acc={val_acc:.4f}")
        print(f"LR: {current_lr:.2e}")
        
        # Early stopping check
        if stopper(val_loss, model, epoch):
            print(f"\nStopping early at epoch {epoch}")
            break
    
    writer.close()
    
    # Load best model
    model.load_state_dict(torch.load(f"{config['checkpoint_dir']}/best.pth"))
    
    print(f"\n{'='*70}")
    print(f"Training Complete!")
    print(f"Best validation loss: {stopper.best_loss:.4f} at epoch {stopper.best_epoch}")
    print(f"Best model saved to: {config['checkpoint_dir']}/best.pth")
    print(f"{'='*70}")
    
    return history, class_names


if __name__ == "__main__":
    history, class_names = train(CONFIG)