"""
compute_stats.py
Compute mean and std for your specific dataset.
Run this once and update config.py with the results.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from dataset import compute_dataset_stats

if __name__ == "__main__":
    print("Computing dataset statistics...")
    print("This may take a few minutes...")
    
    mean, std = compute_dataset_stats('data', img_size=224)
    
    print("\n" + "="*50)
    print("UPDATE YOUR config.py WITH THESE VALUES:")
    print("="*50)
    print(f"\nMEAN = {mean}")
    print(f"STD = {std}")