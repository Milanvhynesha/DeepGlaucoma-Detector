"""
Model architecture for glaucoma detection using DenseNet121.
This module defines the neural network model used for classifying eye fundus images.
"""

import torch
import torch.nn as nn
from monai.networks.nets import DenseNet121

class GlaucomaDetectionModel(nn.Module):
    """
    Glaucoma detection model using DenseNet121 architecture from MONAI.
    
    This model takes RGB eye fundus images (3 channels, 224x224) and outputs
    binary classification: Non-Glaucoma (0) or Glaucoma (1).
    
    Args:
        num_classes (int): Number of output classes (default: 2)
        pretrained (bool): Whether to use pretrained weights (default: True)
    """
    def __init__(self, num_classes=2, pretrained=True):
        super(GlaucomaDetectionModel, self).__init__()
        
        # Use DenseNet121 from MONAI - specifically designed for medical imaging
        self.model = DenseNet121(
            spatial_dims=2,          # 2D images
            in_channels=3,           # RGB images
            out_channels=num_classes,# Binary classification
            pretrained=pretrained    # Use pretrained weights if available
        )
        
        # Store model info for easy access
        self.input_size = (3, 224, 224)
        self.num_classes = num_classes
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, 224, 224)
            
        Returns:
            torch.Tensor: Raw logits for each class (batch_size, num_classes)
        """
        return self.model(x)
    
    def predict_proba(self, x):
        """
        Get probability predictions.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Softmax probabilities for each class
        """
        with torch.no_grad():
            logits = self.forward(x)
            return torch.softmax(logits, dim=1)
    
    def predict(self, x):
        """
        Get class predictions.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Predicted class indices
        """
        with torch.no_grad():
            logits = self.forward(x)
            return torch.argmax(logits, dim=1)


def create_model(device='cuda', pretrained=True):
    """
    Helper function to create and initialize the model.
    
    Args:
        device (str): Device to place model on ('cuda' or 'cpu')
        pretrained (bool): Whether to use pretrained weights
        
    Returns:
        GlaucomaDetectionModel: Initialized model on specified device
    """
    model = GlaucomaDetectionModel(num_classes=2, pretrained=pretrained)
    model = model.to(device)
    return model


def count_parameters(model):
    """
    Count total and trainable parameters in the model.
    
    Args:
        model: PyTorch model
        
    Returns:
        tuple: (total_params, trainable_params)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


if __name__ == "__main__":
    # Quick test when run directly
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = create_model(device)
    print(model)
    
    # Count parameters
    total, trainable = count_parameters(model)
    print(f"\n📊 Model Parameters:")
    print(f"   Total parameters: {total:,}")
    print(f"   Trainable parameters: {trainable:,}")
    
    # Test forward pass
    batch_size = 2
    dummy_input = torch.randn(batch_size, 3, 224, 224).to(device)
    output = model(dummy_input)
    print(f"\n✅ Forward pass successful!")
    print(f"   Input shape: {dummy_input.shape}")
    print(f"   Output shape: {output.shape}")
    
    # Test prediction
    probs = model.predict_proba(dummy_input)
    preds = model.predict(dummy_input)
    print(f"\n📈 Sample predictions:")
    print(f"   Probabilities shape: {probs.shape}")
    print(f"   Predictions: {preds}")
