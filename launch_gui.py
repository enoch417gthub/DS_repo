"""
launch_gui.py
Simple launcher for the Blood Cell Classifier GUI
"""

import sys
import os
import subprocess

def check_model():
    """Check if model exists"""
    model_path = "checkpoints/best.pth"
    if not os.path.exists(model_path):
        print("=" * 60)
        print("⚠️  Model not found!")
        print("=" * 60)
        print(f"Expected model at: {model_path}")
        print("\nPlease train the model first using:")
        print("  python src/train_main.py --epochs 60")
        print("\nOr run a quick training test:")
        print("  python quick_train.py")
        print("=" * 60)
        return False
    return True

if __name__ == "__main__":
    print("🚀 Launching Blood Cell Classifier GUI...")
    
    if check_model():
        try:
            from gui_app import main
            main()
        except ImportError as e:
            print(f"❌ Failed to import GUI: {e}")
            print("\nMake sure you have installed required packages:")
            print("  pip install customtkinter pillow")
    else:
        response = input("\nWould you like to run a quick training first? (y/n): ")
        if response.lower() == 'y':
            subprocess.run([sys.executable, "quick_train.py"])