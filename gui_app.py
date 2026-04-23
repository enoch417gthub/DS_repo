"""
gui_app.py
Modern GUI for Blood Cell Classification
"""

import sys
import os
import threading
from pathlib import Path
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import torch
import numpy as np
import customtkinter as ctk

# Import project modules
from config import CONFIG
from model import BloodCellCNN
from dataset import get_transforms

# Configure CustomTkinter appearance
ctk.set_appearance_mode("dark")  # Modes: "dark", "light", "system"
ctk.set_default_color_theme("blue")  # Themes: "blue", "green", "dark-blue"


class BloodCellClassifierGUI:
    def __init__(self):
        self.root = ctk.CTk()
        self.root.title("Blood Cell Classifier - AI-Powered Diagnosis Assistant")
        self.root.geometry("1280x800")
        self.root.minsize(1024, 700)
        
        # Set icon (optional - create an icon file or remove)
        # self.root.iconbitmap("icon.ico")
        
        # Load model
        self.model = None
        self.transform = None
        self.class_names = CONFIG["class_names"]
        self.device = CONFIG["device"]
        
        # Current image path
        self.current_image_path = None
        self.current_image = None
        
        # Colors
        self.colors = {
            "primary": "#2B5B84",
            "secondary": "#1E3A5F", 
            "success": "#2E7D32",
            "error": "#C62828",
            "warning": "#ED6C02",
            "info": "#0288D1",
            "background": "#0F172A",
            "card_bg": "#1E293B",
            "text": "#F1F5F9",
            "text_secondary": "#94A3B8",
        }
        
        # Load the model in background
        self.load_model()
        
        # Setup UI
        self.setup_ui()
        
        # Bind keyboard shortcuts
        self.setup_shortcuts()
        
    def load_model(self):
        """Load the trained model"""
        try:
            model_path = "checkpoints/best.pth"
            if not os.path.exists(model_path):
                messagebox.showwarning(
                    "Model Not Found",
                    f"Model not found at {model_path}\nPlease train the model first using: python src/train_main.py"
                )
                return False
            
            self.model = BloodCellCNN(
                num_classes=CONFIG["num_classes"],
                base_filters=CONFIG["base_filters"],
                dropout_rate=0.0,  # No dropout for inference
            ).to(self.device)
            
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            
            self.transform = get_transforms(CONFIG["img_size"], mode="val")
            return True
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
            return False
    
    def setup_ui(self):
        """Setup the main UI layout"""
        
        # Main container
        self.main_frame = ctk.CTkFrame(self.root, fg_color=self.colors["background"])
        self.main_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Header
        self.create_header()
        
        # Content area (split into two columns)
        self.content_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.content_frame.pack(fill="both", expand=True, pady=20)
        
        # Left column - Image upload & preview
        self.left_frame = ctk.CTkFrame(
            self.content_frame, 
            fg_color=self.colors["card_bg"],
            corner_radius=15
        )
        self.left_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))
        
        # Right column - Results
        self.right_frame = ctk.CTkFrame(
            self.content_frame,
            fg_color=self.colors["card_bg"],
            corner_radius=15
        )
        self.right_frame.pack(side="right", fill="both", expand=True, padx=(10, 0))
        
        # Setup left panel content
        self.setup_left_panel()
        
        # Setup right panel content
        self.setup_right_panel()
        
        # Footer
        self.create_footer()
    
    def create_header(self):
        """Create the header section"""
        header_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        header_frame.pack(fill="x", pady=(0, 20))
        
        # Title
        title_label = ctk.CTkLabel(
            header_frame,
            text="🩸 Blood Cell Classifier",
            font=ctk.CTkFont(size=32, weight="bold"),
            text_color=self.colors["primary"]
        )
        title_label.pack(side="left")
        
        # Subtitle
        subtitle_label = ctk.CTkLabel(
            header_frame,
            text="AI-Powered Medical Image Analysis",
            font=ctk.CTkFont(size=14),
            text_color=self.colors["text_secondary"]
        )
        subtitle_label.pack(side="left", padx=(15, 0))
        
        # Model status indicator
        status_frame = ctk.CTkFrame(header_frame, fg_color="transparent")
        status_frame.pack(side="right")
        
        status_dot = ctk.CTkLabel(
            status_frame,
            text="●",
            font=ctk.CTkFont(size=16),
            text_color=self.colors["success"] if self.model else self.colors["error"]
        )
        status_dot.pack(side="left")
        
        status_text = ctk.CTkLabel(
            status_frame,
            text="Model Ready" if self.model else "Model Not Loaded",
            font=ctk.CTkFont(size=12),
            text_color=self.colors["text_secondary"]
        )
        status_text.pack(side="left", padx=(5, 0))
    
    def setup_left_panel(self):
        """Setup the left panel with image upload and preview"""
        
        # Upload section
        upload_frame = ctk.CTkFrame(self.left_frame, fg_color="transparent")
        upload_frame.pack(fill="x", padx=20, pady=20)
        
        upload_title = ctk.CTkLabel(
            upload_frame,
            text="📤 Upload Image",
            font=ctk.CTkFont(size=20, weight="bold")
        )
        upload_title.pack(anchor="w")
        
        # Upload buttons
        button_frame = ctk.CTkFrame(upload_frame, fg_color="transparent")
        button_frame.pack(fill="x", pady=(15, 0))
        
        self.upload_btn = ctk.CTkButton(
            button_frame,
            text="📁 Select Image",
            command=self.upload_image,
            height=45,
            font=ctk.CTkFont(size=14),
            fg_color=self.colors["primary"],
            hover_color="#3A7EB6"
        )
        self.upload_btn.pack(side="left", padx=(0, 10))
        
        self.clear_btn = ctk.CTkButton(
            button_frame,
            text="🗑️ Clear",
            command=self.clear_image,
            height=45,
            font=ctk.CTkFont(size=14),
            fg_color="transparent",
            border_width=1,
            border_color=self.colors["text_secondary"],
            hover_color=self.colors["secondary"]
        )
        self.clear_btn.pack(side="left")
        
        # Image preview area
        preview_frame = ctk.CTkFrame(self.left_frame, fg_color=self.colors["background"], corner_radius=10)
        preview_frame.pack(fill="both", expand=True, padx=20, pady=(10, 20))
        
        preview_title = ctk.CTkLabel(
            preview_frame,
            text="🔍 Image Preview",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        preview_title.pack(anchor="w", padx=15, pady=(15, 10))
        
        # Image display area
        self.image_display = ctk.CTkLabel(
            preview_frame,
            text="No image selected\n\nClick 'Select Image' to begin",
            font=ctk.CTkFont(size=14),
            text_color=self.colors["text_secondary"]
        )
        self.image_display.pack(fill="both", expand=True, padx=15, pady=15)
        
        # Image info
        self.image_info = ctk.CTkLabel(
            preview_frame,
            text="",
            font=ctk.CTkFont(size=11),
            text_color=self.colors["text_secondary"]
        )
        self.image_info.pack(anchor="w", padx=15, pady=(0, 15))
    
    def setup_right_panel(self):
        """Setup the right panel with prediction results"""
        
        # Results header
        results_header = ctk.CTkFrame(self.right_frame, fg_color="transparent")
        results_header.pack(fill="x", padx=20, pady=20)
        
        results_title = ctk.CTkLabel(
            results_header,
            text="📊 Analysis Results",
            font=ctk.CTkFont(size=20, weight="bold")
        )
        results_title.pack(anchor="w")
        
        # Prediction result card
        self.result_card = ctk.CTkFrame(
            self.right_frame,
            fg_color=self.colors["background"],
            corner_radius=10
        )
        self.result_card.pack(fill="x", padx=20, pady=(10, 20))
        
        # Predicted class
        self.prediction_label = ctk.CTkLabel(
            self.result_card,
            text="Waiting for image...",
            font=ctk.CTkFont(size=24, weight="bold"),
            text_color=self.colors["text_secondary"]
        )
        self.prediction_label.pack(pady=(20, 10))
        
        # Confidence
        self.confidence_label = ctk.CTkLabel(
            self.result_card,
            text="",
            font=ctk.CTkFont(size=14),
            text_color=self.colors["info"]
        )
        self.confidence_label.pack()
        
        # Confidence bar
        self.confidence_frame = ctk.CTkFrame(self.result_card, fg_color="transparent")
        self.confidence_frame.pack(fill="x", padx=30, pady=15)
        
        self.confidence_bar = ctk.CTkProgressBar(
            self.confidence_frame,
            height=12,
            corner_radius=6
        )
        self.confidence_bar.pack(fill="x")
        self.confidence_bar.set(0)
        
        # All classes probabilities
        probs_title = ctk.CTkLabel(
            self.right_frame,
            text="📈 Class Probabilities",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        probs_title.pack(anchor="w", padx=20, pady=(10, 10))
        
        # Probabilities container
        self.probs_container = ctk.CTkFrame(self.right_frame, fg_color="transparent")
        self.probs_container.pack(fill="both", expand=True, padx=20, pady=(0, 20))
        
        # Create progress bars for each class
        self.progress_bars = {}
        self.progress_labels = {}
        
        class_colors = {
            "eosinophil": "#E15759",
            "lymphocyte": "#4E79A7",
            "monocyte": "#F28E2B",
            "neutrophil": "#76B7B2"
        }
        
        for class_name in self.class_names:
            class_frame = ctk.CTkFrame(self.probs_container, fg_color="transparent")
            class_frame.pack(fill="x", pady=8)
            
            # Class name label
            name_label = ctk.CTkLabel(
                class_frame,
                text=class_name.upper(),
                font=ctk.CTkFont(size=13, weight="bold"),
                width=100
            )
            name_label.pack(side="left", padx=(0, 10))
            
            # Progress bar
            progress = ctk.CTkProgressBar(
                class_frame,
                height=10,
                corner_radius=5,
                progress_color=class_colors.get(class_name, self.colors["primary"])
            )
            progress.pack(side="left", fill="x", expand=True, padx=(0, 10))
            progress.set(0)
            
            # Percentage label
            percent_label = ctk.CTkLabel(
                class_frame,
                text="0%",
                font=ctk.CTkFont(size=12),
                width=45
            )
            percent_label.pack(side="left")
            
            self.progress_bars[class_name] = progress
            self.progress_labels[class_name] = percent_label
    
    def create_footer(self):
        """Create footer section"""
        footer_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        footer_frame.pack(fill="x", pady=(20, 0))
        
        # Separator
        separator = ctk.CTkFrame(footer_frame, height=1, fg_color=self.colors["secondary"])
        separator.pack(fill="x", pady=(0, 15))
        
        # Footer text
        footer_text = ctk.CTkLabel(
            footer_frame,
            text="🔬 AI-Powered Blood Cell Classification | For Research Use Only | Not for Clinical Diagnosis",
            font=ctk.CTkFont(size=11),
            text_color=self.colors["text_secondary"]
        )
        footer_text.pack()
    
    def setup_shortcuts(self):
        """Setup keyboard shortcuts"""
        self.root.bind('<Control-o>', lambda e: self.upload_image())
        self.root.bind('<Control-c>', lambda e: self.clear_image())
    
    def upload_image(self):
        """Open file dialog to select an image"""
        file_path = filedialog.askopenfilename(
            title="Select Blood Cell Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.current_image_path = file_path
            self.display_image(file_path)
            self.predict_image(file_path)
    
    def display_image(self, image_path):
        """Display the selected image in the preview area"""
        try:
            # Load and resize image
            img = Image.open(image_path)
            
            # Store original for info
            self.current_image = img
            
            # Resize for display (maintaining aspect ratio)
            display_size = (450, 450)
            img.thumbnail(display_size, Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(img)
            
            # Update display
            self.image_display.configure(image=photo, text="")
            self.image_display.image = photo  # Keep reference
            
            # Update image info
            file_size = os.path.getsize(image_path) / 1024  # KB
            self.image_info.configure(
                text=f"📐 {img.size[0]} x {img.size[1]} px | 📦 {file_size:.1f} KB | 📁 {os.path.basename(image_path)}"
            )
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to display image: {str(e)}")
    
    def predict_image(self, image_path):
        """Run prediction on the selected image"""
        if not self.model:
            messagebox.showerror("Error", "Model not loaded. Please train the model first.")
            return
        
        try:
            # Load and preprocess image
            img = Image.open(image_path).convert("RGB")
            input_tensor = self.transform(img).unsqueeze(0).to(self.device)
            
            # Run inference
            with torch.no_grad():
                logits = self.model(input_tensor)
                probabilities = torch.softmax(logits, dim=1)[0]
                predicted_class_idx = probabilities.argmax().item()
                confidence = probabilities[predicted_class_idx].item()
            
            predicted_class = self.class_names[predicted_class_idx]
            
            # Update UI with results
            self.update_results(predicted_class, confidence, probabilities.cpu().numpy())
            
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {str(e)}")
    
    def update_results(self, predicted_class, confidence, probabilities):
        """Update the UI with prediction results"""
        
        # Update prediction label with animation effect
        self.prediction_label.configure(
            text=f"🔬 {predicted_class.upper()}",
            text_color=self.colors["success"]
        )
        
        # Update confidence
        self.confidence_label.configure(text=f"Confidence: {confidence * 100:.1f}%")
        self.confidence_bar.set(confidence)
        
        # Update class probabilities
        for i, class_name in enumerate(self.class_names):
            prob = probabilities[i]
            self.progress_bars[class_name].set(prob)
            self.progress_labels[class_name].configure(text=f"{prob * 100:.1f}%")
            
            # Highlight the predicted class
            if class_name == predicted_class:
                self.progress_labels[class_name].configure(
                    text_color=self.colors["success"],
                    font=ctk.CTkFont(size=12, weight="bold")
                )
            else:
                self.progress_labels[class_name].configure(
                    text_color=self.colors["text_secondary"],
                    font=ctk.CTkFont(size=12)
                )
        
        # Add to recent predictions (optional animation)
        self.animate_prediction()
    
    def animate_prediction(self):
        """Simple animation for prediction"""
        # Flash effect on result card
        original_color = self.result_card.cget("fg_color")
        self.result_card.configure(fg_color=self.colors["secondary"])
        self.root.after(200, lambda: self.result_card.configure(fg_color=original_color))
    
    def clear_image(self):
        """Clear the current image and reset results"""
        self.current_image_path = None
        self.current_image = None
        
        # Reset image display
        self.image_display.configure(
            image="",
            text="No image selected\n\nClick 'Select Image' to begin",
            text_color=self.colors["text_secondary"]
        )
        self.image_info.configure(text="")
        
        # Reset results
        self.prediction_label.configure(
            text="Waiting for image...",
            text_color=self.colors["text_secondary"]
        )
        self.confidence_label.configure(text="")
        self.confidence_bar.set(0)
        
        # Reset all progress bars
        for class_name in self.class_names:
            self.progress_bars[class_name].set(0)
            self.progress_labels[class_name].configure(text="0%")
            self.progress_labels[class_name].configure(
                text_color=self.colors["text_secondary"],
                font=ctk.CTkFont(size=12)
            )
    
    def run(self):
        """Run the GUI application"""
        self.root.mainloop()


def main():
    """Main entry point"""
    print("=" * 50)
    print("🩸 Blood Cell Classifier GUI")
    print("=" * 50)
    print("Starting application...")
    print("Please ensure model is trained (checkpoints/best.pth)")
    print("-" * 50)
    
    app = BloodCellClassifierGUI()
    app.run()


if __name__ == "__main__":
    main()