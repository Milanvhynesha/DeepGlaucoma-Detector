"""
Utility functions for glaucoma detection project.
Includes helper functions for plotting, saving/loading, and general utilities.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle
import json
from PIL import Image
import random

def set_seed(seed=42):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"✅ Random seed set to {seed}")


def save_checkpoint(model, optimizer, epoch, loss, filepath):
    """
    Save model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch
        loss: Current loss
        filepath: Path to save checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(checkpoint, filepath)
    print(f"✅ Checkpoint saved to {filepath}")


def load_checkpoint(filepath, model, optimizer=None):
    """
    Load model checkpoint.
    
    Args:
        filepath: Path to checkpoint
        model: Model to load weights into
        optimizer: Optimizer to load state into (optional)
    
    Returns:
        dict: Checkpoint data
    """
    checkpoint = torch.load(filepath, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"✅ Checkpoint loaded from {filepath}")
    return checkpoint


def save_history(history, filepath):
    """
    Save training history to file.
    
    Args:
        history: Training history dictionary
        filepath: Path to save file
    """
    with open(filepath, 'wb') as f:
        pickle.dump(history, f)
    print(f"✅ History saved to {filepath}")


def load_history(filepath):
    """
    Load training history from file.
    
    Args:
        filepath: Path to history file
    
    Returns:
        dict: Training history
    """
    with open(filepath, 'rb') as f:
        history = pickle.load(f)
    print(f"✅ History loaded from {filepath}")
    return history


def save_metadata(metadata, filepath):
    """
    Save metadata as JSON.
    
    Args:
        metadata: Dictionary of metadata
        filepath: Path to save file
    """
    with open(filepath, 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f"✅ Metadata saved to {filepath}")


def get_device():
    """
    Get available device (cuda if available else cpu).
    
    Returns:
        torch.device: Device
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    return device


def count_parameters(model):
    """
    Count number of trainable parameters in model.
    
    Args:
        model: PyTorch model
    
    Returns:
        tuple: (total_params, trainable_params)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n📊 Model Parameters:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Non-trainable parameters: {total_params - trainable_params:,}")
    
    return total_params, trainable_params


def plot_sample_predictions(model, loader, device, num_samples=4, save_path=None):
    """
    Plot sample predictions from dataloader.
    
    Args:
        model: Trained model
        loader: DataLoader
        device: Device to run on
        num_samples: Number of samples to plot
        save_path: Path to save figure
    """
    model.eval()
    
    # Get batch
    images, labels = next(iter(loader))
    images, labels = images.to(device), labels.to(device)
    
    # Get predictions
    with torch.no_grad():
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(outputs, dim=1)
    
    # Plot
    fig, axes = plt.subplots(1, min(num_samples, len(images)), 
                             figsize=(15, 4))
    
    for i in range(min(num_samples, len(images))):
        # Convert tensor to image
        img = images[i].cpu()
        if img.shape[0] == 3:
            img = img.permute(1, 2, 0).numpy()
        
        # Denormalize if needed
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        
        axes[i].imshow(img)
        
        # Set title
        true_label = "G" if labels[i] == 1 else "NG"
        pred_label = "G" if preds[i] == 1 else "NG"
        color = 'green' if preds[i] == labels[i] else 'red'
        
        axes[i].set_title(f"True: {true_label}\nPred: {pred_label}\nProb: {probs[i][1]:.2f}", 
                         color=color)
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def export_to_torchscript(model, filepath, example_input=None):
    """
    Export model to TorchScript for deployment.
    
    Args:
        model: PyTorch model
        filepath: Path to save TorchScript model
        example_input: Example input tensor
    """
    model.eval()
    
    if example_input is None:
        example_input = torch.randn(1, 3, 224, 224)
    
    # Trace model
    traced_model = torch.jit.trace(model, example_input)
    
    # Save
    traced_model.save(filepath)
    print(f"✅ TorchScript model saved to {filepath}")


def get_class_weights(labels):
    """
    Calculate class weights for imbalanced datasets.
    
    Args:
        labels: Array of labels
    
    Returns:
        torch.Tensor: Class weights
    """
    unique, counts = np.unique(labels, return_counts=True)
    n_samples = len(labels)
    n_classes = len(unique)
    
    weights = n_samples / (n_classes * counts)
    
    print(f"Class weights: {dict(zip(unique, weights))}")
    return torch.tensor(weights, dtype=torch.float)


if __name__ == "__main__":
    # Test utilities
    print("🧪 Testing utils.py")
    
    # Test device
    device = get_device()
    
    # Test seed
    set_seed(42)
    
    print("\n✅ All tests passed!")
