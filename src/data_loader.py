"""
Data loading and preprocessing utilities for glaucoma detection.
Handles loading of eye fundus images, creating datasets, and preparing data loaders.
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import pandas as pd
from monai.transforms import (
    Compose, LoadImage, ScaleIntensity, Resize, 
    RandRotate, RandFlip, RandZoom, RandGaussianNoise, ToTensor
)
import torchvision.transforms as T
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class GlaucomaDataset(Dataset):
    """
    Custom dataset for glaucoma eye fundus images.
    
    Args:
        data_path (str): Path to the directory containing images
        labels_df (pd.DataFrame): DataFrame with 'filename' and 'label' columns
        transform (callable, optional): Optional transform to be applied on images
        use_monai (bool): Whether to use MONAI transforms (default: True)
    """
    def __init__(self, data_path, labels_df, transform=None, use_monai=True):
        self.data_path = data_path
        self.labels_df = labels_df
        self.transform = transform
        self.use_monai = use_monai
        self.image_files = []
        self.labels = []
        
        # Match images with labels and verify they exist
        for idx, row in labels_df.iterrows():
            filename = row['filename']
            label = row['label']
            
            # Check if file exists
            full_path = os.path.join(data_path, filename)
            if os.path.exists(full_path):
                self.image_files.append(filename)
                self.labels.append(label)
            else:
                print(f"Warning: Image not found: {filename}")
        
        print(f"✅ Dataset initialized with {len(self.image_files)} valid images")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        """
        Get image and label at specified index.
        
        Returns:
            tuple: (image_tensor, label_tensor)
        """
        img_name = self.image_files[idx]
        img_path = os.path.join(self.data_path, img_name)
        label = self.labels[idx]
        
        try:
            if self.transform and self.use_monai:
                # MONAI transforms (applied to file path)
                image = self.transform(img_path)
                return image, torch.tensor(label, dtype=torch.long)
            
            elif self.transform and not self.use_monai:
                # Torchvision transforms (applied to PIL image)
                image = Image.open(img_path).convert('RGB')
                image = self.transform(image)
                return image, torch.tensor(label, dtype=torch.long)
            
            else:
                # Manual loading (no transforms)
                image = Image.open(img_path).convert('RGB')
                image = image.resize((224, 224))
                image = np.array(image) / 255.0
                image = torch.from_numpy(image).permute(2, 0, 1).float()
                return image, torch.tensor(label, dtype=torch.long)
        
        except Exception as e:
            print(f"❌ Error loading image {img_name}: {e}")
            # Return dummy tensor in case of error
            return torch.zeros((3, 224, 224)), torch.tensor(label, dtype=torch.long)
    
    def get_class_distribution(self):
        """Get count of samples per class."""
        unique, counts = np.unique(self.labels, return_counts=True)
        return dict(zip(unique, counts))
    
    def show_sample(self, idx=None):
        """Display a sample image from the dataset."""
        if idx is None:
            idx = np.random.randint(0, len(self))
        
        img, label = self[idx]
        
        # Convert tensor to displayable image
        if isinstance(img, torch.Tensor):
            if img.shape[0] == 3:  # CHW format
                img_display = img.permute(1, 2, 0).numpy()
            else:
                img_display = img.numpy()
        else:
            img_display = img
        
        # Denormalize if needed
        if img_display.max() <= 1.0:
            img_display = (img_display * 255).astype(np.uint8)
        
        plt.figure(figsize=(6, 6))
        plt.imshow(img_display)
        plt.title(f"Label: {'Glaucoma' if label == 1 else 'Non-Glaucoma'}")
        plt.axis('off')
        plt.show()


def get_monai_transforms(is_train=True):
    """
    Get MONAI transforms for medical imaging.
    
    Args:
        is_train (bool): If True, include data augmentation
    
    Returns:
        Compose: MONAI transform pipeline
    """
    if is_train:
        # Training transforms with augmentation
        return Compose([
            LoadImage(image_only=True),
            ScaleIntensity(),
            Resize((224, 224)),
            RandRotate(range_x=0.2, prob=0.5),
            RandFlip(spatial_axis=0, prob=0.5),
            RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.3),
            RandGaussianNoise(prob=0.3, mean=0.0, std=0.1),
            ToTensor()
        ])
    else:
        # Validation/test transforms (no augmentation)
        return Compose([
            LoadImage(image_only=True),
            ScaleIntensity(),
            Resize((224, 224)),
            ToTensor()
        ])


def get_torchvision_transforms(is_train=True):
    """
    Get torchvision transforms.
    
    Args:
        is_train (bool): If True, include data augmentation
    
    Returns:
        Compose: Torchvision transform pipeline
    """
    if is_train:
        return T.Compose([
            T.Resize((224, 224)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(15),
            T.ColorJitter(brightness=0.1, contrast=0.1),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


def create_dataloaders(data_path, train_df, val_df, test_df, 
                      batch_size=8, num_workers=2, use_monai=True):
    """
    Create train, validation, and test dataloaders.
    
    Args:
        data_path (str): Path to image directory
        train_df (pd.DataFrame): Training labels
        val_df (pd.DataFrame): Validation labels
        test_df (pd.DataFrame): Test labels
        batch_size (int): Batch size for dataloaders
        num_workers (int): Number of worker processes
        use_monai (bool): Whether to use MONAI transforms
    
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    print("\n" + "=" * 50)
    print("CREATING DATALOADERS")
    print("=" * 50)
    
    # Get appropriate transforms
    if use_monai:
        train_transform = get_monai_transforms(is_train=True)
        val_transform = get_monai_transforms(is_train=False)
        print("📊 Using MONAI transforms")
    else:
        train_transform = get_torchvision_transforms(is_train=True)
        val_transform = get_torchvision_transforms(is_train=False)
        print("📊 Using Torchvision transforms")
    
    # Create datasets
    print("\n📁 Creating training dataset...")
    train_dataset = GlaucomaDataset(
        data_path, train_df, 
        transform=train_transform, 
        use_monai=use_monai
    )
    
    print("📁 Creating validation dataset...")
    val_dataset = GlaucomaDataset(
        data_path, val_df, 
        transform=val_transform, 
        use_monai=use_monai
    )
    
    print("📁 Creating test dataset...")
    test_dataset = GlaucomaDataset(
        data_path, test_df, 
        transform=val_transform, 
        use_monai=use_monai
    )
    
    # Print dataset statistics
    print(f"\n📊 Dataset Statistics:")
    print(f"   Training: {len(train_dataset)} images")
    print(f"   Validation: {len(val_dataset)} images")
    print(f"   Test: {len(test_dataset)} images")
    
    print(f"\n📊 Class Distribution (Training):")
    dist = train_dataset.get_class_distribution()
    for cls, count in dist.items():
        class_name = "Glaucoma" if cls == 1 else "Non-Glaucoma"
        print(f"   {class_name}: {count} images ({count/len(train_dataset)*100:.1f}%)")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=False  # Set to True if using GPU
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=False
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=False
    )
    
    print(f"\n✅ Dataloaders created successfully!")
    print(f"   Training batches: {len(train_loader)}")
    print(f"   Validation batches: {len(val_loader)}")
    print(f"   Test batches: {len(test_loader)}")
    
    return train_loader, val_loader, test_loader


def load_labels_from_csv(csv_path, data_path=None):
    """
    Load labels from CSV file.
    
    Args:
        csv_path (str): Path to CSV file
        data_path (str, optional): Path to check if images exist
    
    Returns:
        pd.DataFrame: DataFrame with filename and label columns
    """
    df = pd.read_csv(csv_path)
    
    # Ensure required columns exist
    if 'filename' not in df.columns:
        # Try to find image filename column
        for col in df.columns:
            if 'image' in col.lower() or 'file' in col.lower():
                df = df.rename(columns={col: 'filename'})
                break
    
    if 'label' not in df.columns:
        # Try to find label column
        for col in df.columns:
            if 'label' in col.lower() or 'class' in col.lower():
                df = df.rename(columns={col: 'label'})
                break
    
    # Filter out images that don't exist
    if data_path:
        initial_count = len(df)
        df = df[df['filename'].apply(lambda x: os.path.exists(os.path.join(data_path, x)))]
        print(f"Filtered {initial_count - len(df)} images that don't exist")
    
    return df


def split_data(labels_df, test_size=0.2, val_size=0.2, random_state=42):
    """
    Split data into train, validation, and test sets.
    
    Args:
        labels_df (pd.DataFrame): DataFrame with 'label' column
        test_size (float): Proportion for test set
        val_size (float): Proportion for validation set (from remaining)
        random_state (int): Random seed
    
    Returns:
        tuple: (train_df, val_df, test_df)
    """
    # First split: separate test set
    train_val_df, test_df = train_test_split(
        labels_df, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=labels_df['label']
    )
    
    # Second split: separate validation from training
    train_df, val_df = train_test_split(
        train_val_df, 
        test_size=val_size, 
        random_state=random_state, 
        stratify=train_val_df['label']
    )
    
    print("\n" + "=" * 50)
    print("DATA SPLIT RESULTS")
    print("=" * 50)
    print(f"Total samples: {len(labels_df)}")
    print(f"Training: {len(train_df)} ({len(train_df)/len(labels_df)*100:.1f}%)")
    print(f"Validation: {len(val_df)} ({len(val_df)/len(labels_df)*100:.1f}%)")
    print(f"Test: {len(test_df)} ({len(test_df)/len(labels_df)*100:.1f}%)")
    
    return train_df, val_df, test_df


if __name__ == "__main__":
    # Quick test when run directly
    print("🧪 Testing data_loader.py")
    
    # Create dummy data
    dummy_data = pd.DataFrame({
        'filename': [f"img_{i}.jpg" for i in range(10)],
        'label': [0, 1] * 5
    })
    
    # Test dataset
    dataset = GlaucomaDataset('/dummy/path', dummy_data)
    print(f"\nDataset length: {len(dataset)}")
    print(f"Class distribution: {dataset.get_class_distribution()}")
    
    # Test transforms
    monai_transform = get_monai_transforms(is_train=True)
    tv_transform = get_torchvision_transforms(is_train=True)
    
    print("\n✅ All tests passed!")
