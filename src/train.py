"""
Training utilities for glaucoma detection model.
Contains functions for training loops, validation, and model checkpointing.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import os
import gc
from sklearn.metrics import roc_auc_score

def train_epoch(model, loader, criterion, optimizer, device):
    """
    Train the model for one epoch.
    
    Args:
        model: PyTorch model
        loader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on ('cuda' or 'cpu')
    
    Returns:
        tuple: (epoch_loss, epoch_accuracy)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, labels) in enumerate(loader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Print progress every 10 batches
        if (batch_idx + 1) % 10 == 0:
            print(f'  Batch {batch_idx+1}/{len(loader)} - Loss: {loss.item():.4f}')
    
    epoch_loss = running_loss / len(loader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc


def validate_epoch(model, loader, criterion, device):
    """
    Validate the model for one epoch.
    
    Args:
        model: PyTorch model
        loader: DataLoader for validation data
        criterion: Loss function
        device: Device to run on
    
    Returns:
        tuple: (epoch_loss, epoch_acc, auc, all_labels, all_predictions, all_probs)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of glaucoma class
    
    epoch_loss = running_loss / len(loader)
    epoch_acc = 100 * correct / total
    
    # Calculate AUC
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.0
    
    return epoch_loss, epoch_acc, auc, all_labels, all_predictions, all_probs


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler,
                num_epochs, device, base_path=None, start_epoch=0):
    """
    Complete training loop with checkpointing.
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        num_epochs: Number of epochs to train
        device: Device to train on
        base_path: Path to save model checkpoints
        start_epoch: Starting epoch (for resuming training)
    
    Returns:
        dict: Training history
    """
    # Initialize tracking variables
    best_val_acc = 0.0
    best_val_auc = 0.0
    best_val_loss = float('inf')
    
    # History dictionaries
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'val_auc': [],
        'learning_rates': [],
        'epoch_times': []
    }
    
    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)
    print(f"Total epochs: {num_epochs}")
    print(f"Device: {device}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print("=" * 60)
    
    for epoch in range(start_epoch, num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 40)
        
        # Training
        start_time = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        
        # Validation
        val_loss, val_acc, val_auc, _, _, _ = validate_epoch(model, val_loader, criterion, device)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_auc'].append(val_auc)
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        history['learning_rates'].append(current_lr)
        
        # Update learning rate
        if scheduler:
            scheduler.step(val_loss)
        
        # Calculate time
        epoch_time = time.time() - start_time
        history['epoch_times'].append(epoch_time)
        
        # Print epoch results
        print(f"Time: {epoch_time:.2f}s ({epoch_time/60:.2f} minutes)")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | Val AUC: {val_auc:.4f}")
        print(f"Learning Rate: {current_lr:.6f}")
        
        # Save best models
        if base_path:
            os.makedirs(base_path, exist_ok=True)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_auc': val_auc,
                }, os.path.join(base_path, 'best_model_accuracy.pth'))
                print(f"✓ Saved best accuracy model: {best_val_acc:.2f}%")
            
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                torch.save(model.state_dict(), os.path.join(base_path, 'best_model_auc.pth'))
                print(f"✓ Saved best AUC model: {best_val_auc:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), os.path.join(base_path, 'best_model_loss.pth'))
                print(f"✓ Saved best loss model: {best_val_loss:.4f}")
        
        # Force garbage collection every few epochs
        if (epoch + 1) % 5 == 0:
            gc.collect()
            print("✓ Garbage collection performed")
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETED")
    print("=" * 60)
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Best Validation AUC: {best_val_auc:.4f}")
    
    return history


def resume_training(checkpoint_path, model, optimizer, scheduler=None):
    """
    Resume training from a checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optimizer to load state into
        scheduler: Learning rate scheduler (optional)
    
    Returns:
        tuple: (model, optimizer, start_epoch, best_val_acc)
    """
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    best_val_acc = checkpoint.get('val_acc', 0.0)
    
    print(f"Resuming from epoch {start_epoch} with best validation accuracy: {best_val_acc:.2f}%")
    
    return model, optimizer, start_epoch, best_val_acc
