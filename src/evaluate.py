"""
Evaluation utilities for glaucoma detection model.
Includes metrics, confusion matrix, ROC curves, and classification reports.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, roc_curve, confusion_matrix,
                           classification_report)
import torch
import os

def calculate_metrics(y_true, y_pred, y_prob=None):
    """
    Calculate comprehensive classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Prediction probabilities (for AUC)
    
    Returns:
        dict: Dictionary of metrics
    """
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, average='binary')
    metrics['recall'] = recall_score(y_true, y_pred, average='binary')
    metrics['f1_score'] = f1_score(y_true, y_pred, average='binary')
    
    # AUC if probabilities provided
    if y_prob is not None:
        metrics['auc'] = roc_auc_score(y_true, y_prob)
    
    return metrics


def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        save_path: Path to save figure (optional)
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Non-Glaucoma', 'Glaucoma'],
                yticklabels=['Non-Glaucoma', 'Glaucoma'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    
    # Add percentages
    total = np.sum(cm)
    for i in range(2):
        for j in range(2):
            plt.text(j+0.5, i+0.7, f'({cm[i,j]/total*100:.1f}%)',
                    ha='center', va='center', color='red', fontsize=10)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def plot_roc_curve(y_true, y_prob, save_path=None):
    """
    Plot ROC curve.
    
    Args:
        y_true: True labels
        y_prob: Prediction probabilities for positive class
        save_path: Path to save figure (optional)
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)
    
    # Find optimal threshold
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
             label='Random Classifier')
    plt.plot(fpr[optimal_idx], tpr[optimal_idx], 'ro',
             label=f'Optimal Threshold\n(Value: {optimal_threshold:.3f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()
    
    return optimal_threshold


def plot_training_history(history, save_path=None):
    """
    Plot training history curves.
    
    Args:
        history: Dictionary containing training history
        save_path: Path to save figure (optional)
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Loss curves
    axes[0, 0].plot(history['train_loss'], label='Training Loss', color='blue')
    axes[0, 0].plot(history['val_loss'], label='Validation Loss', color='red')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy curves
    axes[0, 1].plot(history['train_acc'], label='Training Accuracy', color='blue')
    axes[0, 1].plot(history['val_acc'], label='Validation Accuracy', color='red')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].set_title('Training and Validation Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # AUC curve
    axes[1, 0].plot(history['val_auc'], label='Validation AUC', color='green')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('AUC')
    axes[1, 0].set_title('Validation AUC Over Time')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Learning rate
    axes[1, 1].plot(history['learning_rates'], label='Learning Rate', color='purple')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].set_title('Learning Rate Schedule')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_yscale('log')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def evaluate_model(model, test_loader, criterion, device, save_path=None):
    """
    Complete model evaluation on test set.
    
    Args:
        model: Trained PyTorch model
        test_loader: Test data loader
        criterion: Loss function
        device: Device to run on
        save_path: Path to save plots
    
    Returns:
        dict: Evaluation results
    """
    from src.train import validate_epoch
    
    # Get predictions
    test_loss, test_acc, test_auc, y_true, y_pred, y_prob = validate_epoch(
        model, test_loader, criterion, device
    )
    
    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred, y_prob)
    
    print("\n" + "=" * 60)
    print("TEST SET EVALUATION")
    print("=" * 60)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.2f}%")
    print(f"Test AUC: {test_auc:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_true, y_pred, 
                               target_names=['Non-Glaucoma', 'Glaucoma']))
    
    # Plot confusion matrix
    if save_path:
        cm_path = os.path.join(save_path, 'confusion_matrix.png')
        plot_confusion_matrix(y_true, y_pred, cm_path)
    else:
        plot_confusion_matrix(y_true, y_pred)
    
    # Plot ROC curve
    if save_path:
        roc_path = os.path.join(save_path, 'roc_curve.png')
        plot_roc_curve(y_true, y_prob, roc_path)
    else:
        plot_roc_curve(y_true, y_prob)
    
    return {
        'loss': test_loss,
        'accuracy': test_acc,
        'auc': test_auc,
        'y_true': y_true,
        'y_pred': y_pred,
        'y_prob': y_prob,
        'metrics': metrics
    }
