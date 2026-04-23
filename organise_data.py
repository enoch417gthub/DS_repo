"""
organise_data.py
Reorganise raw data into train/val/test splits with proper folder structure.
Run this BEFORE training.
"""

import os
import shutil
import random
from sklearn.model_selection import train_test_split

def organise_dataset():
    """
    Reorganise data from the existing structure into:
    data/
        train/
            eosinophil/
            lymphocyte/
            monocyte/
            neutrophil/
        val/
            same subfolders
        test/
            same subfolders
    """
    
    # Source paths (based on your folder structure)
    source_base = "data/dataset2-master/dataset2-master/images"
    
    # Training data source
    train_source = os.path.join(source_base, "TRAIN")
    
    # Test data source (use TEST_SIMPLE or TEST - I'll use TEST_SIMPLE as it's likely cleaner)
    test_source = os.path.join(source_base, "TEST_SIMPLE")
    
    # Destination paths
    dest_base = "data"
    train_dest = os.path.join(dest_base, "train")
    val_dest = os.path.join(dest_base, "val")
    test_dest = os.path.join(dest_base, "test")
    
    # Class names (as they appear in folders)
    classes = ["EOSINOPHIL", "LYMPHOCYTE", "MONOCYTE", "NEUTROPHIL"]
    # Lowercase for our config
    class_names_lower = ["eosinophil", "lymphocyte", "monocyte", "neutrophil"]
    
    # Create destination directories
    for dest in [train_dest, val_dest, test_dest]:
        for class_name in class_names_lower:
            os.makedirs(os.path.join(dest, class_name), exist_ok=True)
    
    # Process training data: split into train (80%) and val (20%)
    print("Processing training data...")
    for class_idx, class_name in enumerate(classes):
        class_lower = class_names_lower[class_idx]
        source_dir = os.path.join(train_source, class_name)
        
        if not os.path.exists(source_dir):
            print(f"Warning: {source_dir} not found")
            continue
        
        # Get all image files
        images = [f for f in os.listdir(source_dir) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        print(f"  {class_name}: {len(images)} images")
        
        # Split into train (80%) and val (20%)
        train_images, val_images = train_test_split(
            images, test_size=0.2, random_state=42
        )
        
        # Copy to train folder
        for img in train_images:
            src = os.path.join(source_dir, img)
            dst = os.path.join(train_dest, class_lower, img)
            shutil.copy2(src, dst)
        
        # Copy to val folder
        for img in val_images:
            src = os.path.join(source_dir, img)
            dst = os.path.join(val_dest, class_lower, img)
            shutil.copy2(src, dst)
    
    # Process test data
    print("\nProcessing test data...")
    for class_idx, class_name in enumerate(classes):
        class_lower = class_names_lower[class_idx]
        source_dir = os.path.join(test_source, class_name)
        
        if not os.path.exists(source_dir):
            print(f"Warning: {source_dir} not found")
            # Try the TEST folder instead
            source_dir = os.path.join(source_base, "TEST", class_name)
            if not os.path.exists(source_dir):
                continue
        
        images = [f for f in os.listdir(source_dir) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        print(f"  {class_name}: {len(images)} test images")
        
        # Copy all to test folder
        for img in images:
            src = os.path.join(source_dir, img)
            dst = os.path.join(test_dest, class_lower, img)
            shutil.copy2(src, dst)
    
    # Print summary
    print("\n" + "="*50)
    print("DATA ORGANISATION COMPLETE")
    print("="*50)
    
    for split in ["train", "val", "test"]:
        print(f"\n{split.upper()} set:")
        for class_name in class_names_lower:
            path = os.path.join(dest_base, split, class_name)
            if os.path.exists(path):
                count = len(os.listdir(path))
                print(f"  {class_name}: {count} images")
            else:
                print(f"  {class_name}: 0 images")


if __name__ == "__main__":
    organise_dataset()