"""
quick_train.py
Quick training for testing - runs fewer epochs to verify everything works.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from config import CONFIG
from train import train
from visualise import plot_training_history, print_metrics_table

if __name__ == "__main__":
    print("="*70)
    print("QUICK TRAINING TEST (10 epochs)")
    print("="*70)
    
    # Override for quick test
    CONFIG['num_epochs'] = 10
    CONFIG['warmup_epochs'] = 2
    
    print(f"Device: {CONFIG['device']}")
    print(f"Epochs: {CONFIG['num_epochs']}")
    print(f"Batch size: {CONFIG['batch_size']}")
    print(f"Learning rate: {CONFIG['lr']}")
    
    try:
        history, class_names = train(CONFIG)
        
        print("\n" + "="*70)
        print("TRAINING COMPLETE - Quick test successful!")
        print("="*70)
        
        print_metrics_table(history)
        
        # Quick plot
        plot_training_history(history, save_path="quick_test_curves.png")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()