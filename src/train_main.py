"""
train_main.py
Main entry point for training the Blood Cell CNN.
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import CONFIG
from train import train
from visualise import plot_training_history, print_metrics_table
from evaluate import full_evaluation
from dataset import get_dataloaders
from sanity_check import sanity_check


def main():
    parser = argparse.ArgumentParser(description='Train Blood Cell CNN')
    parser.add_argument('--skip_sanity', action='store_true', 
                        help='Skip sanity check before training')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override number of epochs')
    parser.add_argument('--lr', type=float, default=None,
                        help='Override learning rate')
    
    args = parser.parse_args()
    
    # Override config if specified
    if args.epochs:
        CONFIG['num_epochs'] = args.epochs
        print(f"Overriding epochs to {args.epochs}")
    
    if args.lr:
        CONFIG['lr'] = args.lr
        print(f"Overriding learning rate to {args.lr}")
    
    # Run sanity check first
    if not args.skip_sanity:
        print("\n" + "!"*70)
        print("RUNNING SANITY CHECK...")
        print("!"*70)
        
        if not sanity_check():
            print("\n❌ Sanity check failed. Fix issues before training.")
            print("   Use --skip_sanity to bypass (not recommended)")
            return
    
    # Train the model
    print("\n" + "🚀"*35)
    print("STARTING TRAINING")
    print("🚀"*35)
    
    history, class_names = train(CONFIG)
    
    # Plot training history
    print("\n" + "📊"*35)
    print("VISUALISING RESULTS")
    print("📊"*35)
    
    plot_training_history(history, save_path=f"{CONFIG['checkpoint_dir']}/training_curves.png")
    print_metrics_table(history)
    
    # Evaluate on validation set
    print("\n" + "📈"*35)
    print("FINAL VALIDATION EVALUATION")
    print("📈"*35)
    
    loaders, class_names = get_dataloaders(CONFIG)
    
    from model import BloodCellCNN
    model = BloodCellCNN(
        num_classes=CONFIG["num_classes"],
        base_filters=CONFIG["base_filters"],
        dropout_rate=CONFIG["dropout_rate"],
    ).to(CONFIG["device"])
    
    best_model_path = f"{CONFIG['checkpoint_dir']}/best.pth"
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=CONFIG["device"]))
        print(f"\nLoaded best model from {best_model_path}")
    else:
        print(f"\nWarning: Best model not found at {best_model_path}")
    
    full_evaluation(model, loaders["val"], CONFIG["device"], class_names, 
                    save_dir=CONFIG["checkpoint_dir"])
    
    print("\n" + "🎉"*35)
    print("TRAINING COMPLETE!")
    print(f"Best model saved to: {CONFIG['checkpoint_dir']}/best.pth")
    print(f"Training curves saved to: {CONFIG['checkpoint_dir']}/training_curves.png")
    print("🎉"*35)


if __name__ == "__main__":
    import torch
    main()