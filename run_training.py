"""
run_training.py
Launch training with optional arguments.
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip_sanity', action='store_true')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    
    args = parser.parse_args()
    
    cmd = f"python src/train_main.py"
    if args.skip_sanity:
        cmd += " --skip_sanity"
    if args.epochs:
        cmd += f" --epochs {args.epochs}"
    if args.lr:
        cmd += f" --lr {args.lr}"
    
    print(f"Running: {cmd}")
    os.system(cmd)