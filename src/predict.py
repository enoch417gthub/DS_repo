"""
predict.py
Inference on single images and batch prediction on folders.
"""

import torch
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from config import CONFIG, MEAN, STD
from model import BloodCellCNN
from dataset import get_transforms


def predict_single(image_path, model, transform, class_names, device):
    """
    Predict the class of a single blood cell image.
    
    Args:
        image_path: path to image file
        model: PyTorch model
        transform: image transformation pipeline
        class_names: list of class names
        device: 'cuda' or 'cpu'
    
    Returns:
        predicted_class: string
        probabilities: numpy array
    """
    model.eval()
    
    # Load and transform image
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)  # add batch dimension
    
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
        pred_idx = probs.argmax().item()
        confidence = probs[pred_idx].item()
    
    predicted_class = class_names[pred_idx]
    
    # Print results
    print(f"\n{'='*50}")
    print(f"Image: {os.path.basename(image_path)}")
    print(f"{'='*50}")
    print(f"Predicted: {predicted_class}")
    print(f"Confidence: {confidence*100:.1f}%")
    print(f"\nAll class probabilities:")
    print("-" * 30)
    
    for name, p in zip(class_names, probs):
        bar = "#" * int(p.item() * 30)
        print(f"{name:15s} {p.item()*100:5.1f}% {bar}")
    
    return predicted_class, probs.cpu().numpy()


class UnlabelledImageDataset(Dataset):
    """Load images from a flat folder with no labels."""
    
    def __init__(self, folder, transform):
        self.paths = []
        for f in sorted(os.listdir(folder)):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                self.paths.append(os.path.join(folder, f))
        self.transform = transform
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img), self.paths[idx]


def batch_predict(folder, model, transform, class_names, device, 
                  output_csv="predictions.csv", batch_size=32):
    """
    Run inference on all images in a folder.
    
    Args:
        folder: path to folder containing images
        model: PyTorch model
        transform: image transformation pipeline
        class_names: list of class names
        device: 'cuda' or 'cpu'
        output_csv: path to save predictions CSV
        batch_size: batch size for inference
    
    Returns:
        df: pandas DataFrame with predictions
    """
    model.eval()
    
    # Create dataset and loader
    dataset = UnlabelledImageDataset(folder, transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, 
                        num_workers=2)
    
    rows = []
    
    print(f"\nRunning batch prediction on {len(dataset)} images...")
    
    with torch.no_grad():
        for imgs, paths in loader:
            logits = model(imgs.to(device))
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = probs.argmax(axis=1)
            
            for path, pred_idx, prob_arr in zip(paths, preds, probs):
                rows.append({
                    "filename": os.path.basename(path),
                    "prediction": class_names[pred_idx],
                    "confidence": prob_arr[pred_idx],
                    **{f"prob_{class_names[i]}": prob_arr[i] 
                       for i in range(len(class_names))}
                })
    
    # Create DataFrame and save
    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    
    print(f"\nPredictions saved to {output_csv}")
    print(f"\nFirst 5 predictions:")
    print(df.head().to_string())
    
    # Summary
    print(f"\nPrediction summary:")
    print(df['prediction'].value_counts())
    
    return df


def load_model_for_inference(model_path, config):
    """
    Load a trained model for inference.
    
    Args:
        model_path: path to saved model weights (.pth file)
        config: configuration dictionary
    
    Returns:
        model: loaded model in eval mode
        transform: validation transform
        class_names: list of class names
    """
    device = config["device"]
    
    model = BloodCellCNN(
        num_classes=config["num_classes"],
        base_filters=config["base_filters"],
        dropout_rate=config["dropout_rate"],
    ).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    transform = get_transforms(config["img_size"], mode="val")
    
    print(f"Model loaded from {model_path}")
    print(f"Device: {device}")
    
    return model, transform, config["class_names"]


if __name__ == "__main__":
    # Example usage
    model_path = "checkpoints/best.pth"
    
    if os.path.exists(model_path):
        model, transform, class_names = load_model_for_inference(model_path, CONFIG)
        
        # Single image prediction
        # predict_single("test_image.jpg", model, transform, class_names, CONFIG["device"])
        
        # Batch prediction on a folder
        # batch_predict("data/test_images", model, transform, class_names, CONFIG["device"])
    else:
        print(f"Model not found at {model_path}. Train the model first.")