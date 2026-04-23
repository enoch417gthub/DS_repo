"""
visualise.py
Training visualisation: loss curves, predictions, confidence distributions.
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
from PIL import Image


def plot_training_history(history, save_path="training_curves.png"):
    """
    Create comprehensive training visualisation.
    
    Args:
        history: dict with keys: train_loss, val_loss, train_acc, val_acc, lr
        save_path: where to save the figure
    """
    epochs = range(1, len(history["train_loss"]) + 1)
    best_val_epoch = int(np.argmin(history["val_loss"])) + 1
    
    fig = plt.figure(figsize=(18, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)
    
    # Panel 1: Loss curves
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(epochs, history["train_loss"], 'b-o', markersize=4, linewidth=1.8, label='Train loss')
    ax1.plot(epochs, history["val_loss"], 'r-o', markersize=4, linewidth=1.8, label='Val loss')
    ax1.axvline(best_val_epoch, color='green', linestyle='--', linewidth=1.5, 
                label=f"Best val (epoch {best_val_epoch})")
    ax1.set_title('Training & Validation Loss', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Cross-Entropy Loss')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Panel 2: Accuracy curves
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(epochs, [a*100 for a in history["train_acc"]], 'b-o', markersize=4, 
             linewidth=1.8, label='Train acc')
    ax2.plot(epochs, [a*100 for a in history["val_acc"]], 'r-o', markersize=4, 
             linewidth=1.8, label='Val acc')
    ax2.axvline(best_val_epoch, color='green', linestyle='--', linewidth=1.5)
    ax2.set_title('Training & Validation Accuracy', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_ylim(0, 105)
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # Panel 3: Generalisation gap (overfitting indicator)
    ax3 = fig.add_subplot(gs[1, 0])
    gap = [v - t for t, v in zip(history["train_loss"], history["val_loss"])]
    ax3.plot(epochs, gap, 'purple', linewidth=2, label='Val loss - Train loss')
    ax3.axhline(0, color='k', linewidth=0.8, linestyle='--')
    ax3.fill_between(epochs, 0, gap, where=[g > 0 for g in gap], 
                      alpha=0.2, color='red', label='Overfitting region')
    ax3.set_title('Generalisation Gap', fontsize=13, fontweight='bold')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss gap')
    ax3.legend()
    ax3.grid(alpha=0.3)
    
    # Panel 4: Learning rate schedule
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(epochs, history["lr"], 'darkorange', linewidth=2)
    ax4.set_title('Learning Rate Schedule', fontsize=13, fontweight='bold')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Learning Rate')
    ax4.set_yscale('log')
    ax4.grid(which='both', alpha=0.3)
    
    # Summary annotation
    best_acc = max(history["val_acc"]) * 100
    best_loss = min(history["val_loss"])
    fig.suptitle(
        f"Blood Cell CNN Training Summary | "
        f"Best val acc: {best_acc:.2f}% | "
        f"Best val loss: {best_loss:.4f} | "
        f"Best epoch: {best_val_epoch}",
        fontsize=14, fontweight='bold', y=1.02
    )
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Training curves saved to {save_path}")


def print_metrics_table(history):
    """
    Print a clean table of all epoch metrics.
    """
    header = f"{'Epoch':>6} | {'Train Loss':>10} | {'Val Loss':>8} | {'Train Acc':>9} | {'Val Acc':>7} | {'LR':>10}"
    print("\n" + header)
    print("-" * len(header))
    
    best_val_idx = np.argmin(history["val_loss"])
    
    for i in range(len(history["train_loss"])):
        mark = "  <-- BEST" if i == best_val_idx else ""
        print(f"{i+1:>6} | {history['train_loss'][i]:>10.4f} | "
              f"{history['val_loss'][i]:>8.4f} | "
              f"{history['train_acc'][i]*100:>8.2f}% | "
              f"{history['val_acc'][i]*100:>6.2f}% | "
              f"{history['lr'][i]:>10.2e}{mark}")


def visualise_predictions(model, loader, class_names, device, 
                          n_images=16, save_path="predictions.png"):
    """
    Show a grid of images with predicted and true labels.
    Green border = correct, red border = wrong.
    """
    model.eval()
    images_shown = []
    preds_shown = []
    labels_shown = []
    confs_shown = []
    
    # Get mean and std for denormalisation
    from config import MEAN, STD
    mean = np.array(MEAN)
    std = np.array(STD)
    
    with torch.no_grad():
        for imgs, labels in loader:
            logits = model(imgs.to(device))
            probs = torch.softmax(logits, dim=1)
            conf, pred = probs.max(1)
            
            for i in range(len(imgs)):
                if len(images_shown) >= n_images:
                    break
                images_shown.append(imgs[i])
                preds_shown.append(pred[i].item())
                labels_shown.append(labels[i].item())
                confs_shown.append(conf[i].item())
            
            if len(images_shown) >= n_images:
                break
    
    ncols = 4
    nrows = (n_images + 3) // 4
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*3.5, nrows*3.5))
    axes = axes.flat
    
    for ax, img_t, pred, label, conf in zip(axes, images_shown, preds_shown, 
                                             labels_shown, confs_shown):
        # Denormalise
        arr = img_t.numpy().transpose(1, 2, 0)
        arr = np.clip(arr * std + mean, 0, 1)
        ax.imshow(arr)
        
        correct = (pred == label)
        color = "green" if correct else "red"
        
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(3)
        
        ax.set_title(
            f"Pred: {class_names[pred]} ({conf*100:.0f}%)\nTrue: {class_names[label]}",
            fontsize=9, color=color
        )
        ax.axis("off")
    
    # Turn off unused axes
    for ax in list(axes)[len(images_shown):]:
        ax.axis("off")
    
    plt.suptitle("Model Predictions (green=correct, red=wrong)", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches='tight')
    plt.show()


def confidence_distribution(model, loader, device, save_path="confidence.png"):
    """
    Plot confidence distribution for correct vs wrong predictions.
    A good model is confident when correct and uncertain when wrong.
    """
    model.eval()
    correct_confs = []
    wrong_confs = []
    
    with torch.no_grad():
        for imgs, labels in loader:
            logits = model(imgs.to(device))
            probs = torch.softmax(logits, dim=1)
            conf, pred = probs.max(1)
            
            for c, p, l in zip(conf.cpu(), pred.cpu(), labels):
                if p == l:
                    correct_confs.append(c.item())
                else:
                    wrong_confs.append(c.item())
    
    plt.figure(figsize=(10, 5))
    bins = np.linspace(0, 1, 25)
    
    plt.hist(correct_confs, bins=bins, alpha=0.7, color='green', 
             label=f"Correct ({len(correct_confs)} samples)")
    plt.hist(wrong_confs, bins=bins, alpha=0.7, color='red', 
             label=f"Wrong ({len(wrong_confs)} samples)")
    
    plt.xlabel("Confidence (max softmax probability)")
    plt.ylabel("Count")
    plt.title("Confidence Distribution: Correct vs Wrong Predictions")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.show()
    
    print(f"Mean confidence (correct): {np.mean(correct_confs):.3f}")
    print(f"Mean confidence (wrong): {np.mean(wrong_confs):.3f}")


def plot_class_distribution(data_dir="data"):
    """
    Plot class distribution for train/val/test sets.
    """
    import os
    
    classes = ['eosinophil', 'lymphocyte', 'monocyte', 'neutrophil']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, split in enumerate(['train', 'val', 'test']):
        counts = []
        for class_name in classes:
            class_dir = os.path.join(data_dir, split, class_name)
            if os.path.exists(class_dir):
                counts.append(len(os.listdir(class_dir)))
            else:
                counts.append(0)
        
        axes[idx].bar(classes, counts, color=['#4e79a7', '#f28e2b', '#e15759', '#76b7b2'])
        axes[idx].set_title(f'{split.upper()} Set', fontweight='bold')
        axes[idx].set_ylabel('Number of images')
        axes[idx].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for i, v in enumerate(counts):
            axes[idx].text(i, v + 10, str(v), ha='center', fontweight='bold')
    
    plt.suptitle('Dataset Class Distribution', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('class_distribution.png', dpi=120, bbox_inches='tight')
    plt.show()