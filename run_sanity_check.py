
"""
run_sanity_check.py
Quick script to verify everything is working.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from sanity_check import sanity_check

if __name__ == "__main__":
    print("\n" + "="*70)
    print("RUNNING COMPLETE SANITY CHECK")
    print("="*70)
    
    passed = sanity_check()
    
    if passed:
        print("\n✅ Everything looks good! Ready to train.")
        print("\nTo start training, run:")
        print("    python src/train_main.py")
    else:
        print("\n❌ Please fix the issues above before training.")