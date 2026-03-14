"""
Prediction utilities for glaucoma detection.
Functions for making predictions on single images and batches.
"""

import torch
import numpy as np
from PIL import Image
import os
import pandas as pd
from monai.transforms import Compose, LoadImage, ScaleIntensity, Resize, ToTensor
import torchvision.transforms as T
import matplotlib.pyplot as plt

class GlaucomaPredictor:
    """
    Predictor class for glaucoma detection.
    
    Args:
        model: Trained PyTorch model
        device: Device to run inference on ('cuda' or 'cpu')
        use_monai: Whether to use MONAI transforms
    """
    def __init__(self, model, device='cpu', use_monai=True):
        self.model = model
        self.device = device
        self.use_monai = use_monai
        self.model.eval()
        self.model.to(device)
        
        # Setup transforms
        if use_monai:
            self.transform = Compose([
                LoadImage(image_only=True),
                ScaleIntensity(),
                Resize((224, 224)),
                ToTensor()
            ])
        else:
            self.transform = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    def preprocess_image(self, image_path):
        """
        Preprocess a single image for prediction.
        
        Args:
            image_path: Path to image file
            
        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        if self.use_monai:
            # MONAI transform
            image = self.transform(image_path)
            return image.unsqueeze(0).to(self.device)
        else:
            # Torchvision transform
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image)
            return image.unsqueeze(0).to(self.device)
    
    def predict(self, image_path):
        """
        Predict on a single image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            dict: Prediction results
        """
        try:
            # Preprocess image
            image_tensor = self.preprocess_image(image_path)
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_class = torch.argmax(outputs, dim=1).item()
                confidence = probabilities[0][predicted_class].item()
            
            # Prepare result
            result = {
                'prediction': 'Glaucoma' if predicted_class == 1 else 'Non-Glaucoma',
                'confidence': confidence,
                'probabilities': {
                    'Non-Glaucoma': probabilities[0][0].item(),
                    'Glaucoma': probabilities[0][1].item()
                },
                'class': predicted_class,
                'image_path': image_path
            }
            
            return result
        
        except Exception as e:
            print(f"Error predicting on {image_path}: {e}")
            return None
    
    def predict_batch(self, image_paths, show_progress=True):
        """
        Predict on multiple images.
        
        Args:
            image_paths: List of image paths
            show_progress: Whether to show progress bar
            
        Returns:
            pd.DataFrame: Results dataframe
        """
        results = []
        
        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(image_paths, desc="Predicting")
        else:
            iterator = image_paths
        
        for img_path in iterator:
            result = self.predict(img_path)
            if result:
                results.append({
                    'image': os.path.basename(img_path),
                    'prediction': result['prediction'],
                    'confidence': result['confidence'],
                    'prob_non_glaucoma': result['probabilities']['Non-Glaucoma'],
                    'prob_glaucoma': result['probabilities']['Glaucoma'],
                    'class': result['class']
                })
        
        return pd.DataFrame(results)
    
    def predict_from_folder(self, folder_path, extensions=None):
        """
        Predict on all images in a folder.
        
        Args:
            folder_path: Path to folder containing images
            extensions: List of image extensions to include
            
        Returns:
            pd.DataFrame: Results dataframe
        """
        if extensions is None:
            extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
        
        image_paths = []
        for ext in extensions:
            image_paths.extend(glob.glob(os.path.join(folder_path, f'*{ext}')))
        
        print(f"Found {len(image_paths)} images")
        return self.predict_batch(image_paths)
    
    def visualize_prediction(self, image_path, save_path=None):
        """
        Predict and visualize result with image.
        
        Args:
            image_path: Path to image file
            save_path: Path to save visualization (optional)
        """
        result = self.predict(image_path)
        
        if result:
            plt.figure(figsize=(10, 8))
            
            # Load and display image
            img = Image.open(image_path).convert('RGB')
            plt.imshow(img)
            
            # Set title
            color = 'red' if result['class'] == 1 else 'green'
            title = f"Prediction: {result['prediction']}\n"
            title += f"Confidence: {result['confidence']:.2%}\n"
            title += f"Glaucoma: {result['probabilities']['Glaucoma']:.2%} | "
            title += f"Non-Glaucoma: {result['probabilities']['Non-Glaucoma']:.2%}"
            
            plt.title(title, color=color, fontsize=12, fontweight='bold')
            plt.axis('off')
            
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"Saved visualization to {save_path}")
            
            plt.show()
        
        return result


def predict_glaucoma(image_path, model, device='cpu', use_monai=True):
    """
    Simple function for quick prediction.
    
    Args:
        image_path: Path to image file
        model: Trained PyTorch model
        device: Device to run on
        use_monai: Whether to use MONAI transforms
        
    Returns:
        dict: Prediction results
    """
    predictor = GlaucomaPredictor(model, device, use_monai)
    return predictor.predict(image_path)
