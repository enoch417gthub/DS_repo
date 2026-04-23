"""
evaluate.py
Comprehensive evaluation metrics: classification report, confusion matrix, etc.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from tqdm import tqdm


def full_evaluation(model, loader, device, class_names, save_dir="checkpoints"):
    """
    Compute and display all evaluation metrics.
    
    Args:
        model: PyTorch model
        loader: DataLoader for test/val set
        device: 'cuda' or 'cpu'
        class_names: list of class names
        save_dir: directory to save plots
    
    Returns:
        all_preds: numpy array of predictions
        all_labels: numpy array of true labels
        all_probs: numpy array of probabilities
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    print("\nRunning evaluation...")
    with torch.no_grad():
        pbar = tqdm(loader, desc="Evaluating")
        for imgs, labels in pbar:
            imgs = imgs.to(device)
            logits = model(imgs)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # 1. Classification report
    print("\n" + "="*70)
    print("CLASSIFICATION REPORT")
    print("="*70)
    print(classification_report(
        all_labels, all_preds, 
        target_names=class_names, 
        digits=4
    ))
    
    # Calculate macro F1
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    weighted_f1 = f1_score(all_labels, all_preds, average='weighted')
    print(f"\nMacro F1-score: {macro_f1:.4f}")
    print(f"Weighted F1-score: {weighted_f1:.4f}")
    
    # 2. Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=ax1, linewidths=0.5)
    ax1.set_title('Confusion Matrix (counts)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('True Label', fontsize=12)
    ax1.set_xlabel('Predicted Label', fontsize=12)
    
    # Percentages
    sns.heatmap(cm_pct, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=ax2, linewidths=0.5, vmin=0, vmax=100)
    ax2.set_title('Confusion Matrix (% per true class)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('True Label', fontsize=12)
    ax2.set_xlabel('Predicted Label', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/confusion_matrix.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    # 3. Per-class accuracy
    print("\n" + "="*70)
    print("PER-CLASS ACCURACY")
    print("="*70)
    for i, class_name in enumerate(class_names):
        class_mask = (all_labels == i)
        class_correct = (all_preds[class_mask] == all_labels[class_mask]).sum()
        class_total = class_mask.sum()
        if class_total > 0:
            acc = class_correct / class_total
            print(f"{class_name:15s}: {acc:.4f} ({class_correct}/{class_total})")
    
    return all_preds, all_labels, all_probs


def evaluate_test_set(model_path, config, loaders, class_names):
    """
    Load a saved model and evaluate on test set.
    Use this ONLY once at the end of the project.
    
    Args:
        model_path: path to saved model weights
        config: configuration dictionary
        loaders: dictionary of dataloaders
        class_names: list of class names
    """
    from model import BloodCellCNN
    
    device = config["device"]
    
    # Load model
    model = BloodCellCNN(
        num_classes=config["num_classes"],
        base_filters=config["base_filters"],
        dropout_rate=config["dropout_rate"],
    ).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Loaded model from {model_path}")
    
    # Evaluate on test set
    print("\n" + "!"*70)
    print("FINAL TEST SET EVALUATION")
    print("!"*70)
    
    preds, labels, probs = full_evaluation(
        model, loaders["test"], device, class_names, config["checkpoint_dir"]
    )
    
    return preds, labels, probs


if __name__ == "__main__":
    from config import CONFIG
    from dataset import get_dataloaders
    from model import BloodCellCNN
    
    # Load data
    loaders, class_names = get_dataloaders(CONFIG)
    
    # Load best model
    model = BloodCellCNN(
        num_classes=CONFIG["num_classes"],
        base_filters=CONFIG["base_filters"],
        dropout_rate=CONFIG["dropout_rate"],
    ).to(CONFIG["device"])
    
    best_model_path = f"{CONFIG['checkpoint_dir']}/best.pth"
    model.load_state_dict(torch.load(best_model_path, map_location=CONFIG["device"]))
    
    # Evaluate on validation set
    print("\n" + "="*70)
    print("VALIDATION SET EVALUATION")
    print("="*70)
    full_evaluation(model, loaders["val"], CONFIG["device"], class_names)