#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Wildlife species classification module for AI for Sustainable Ecosystems project.
This script implements deep learning models for identifying animal species from images.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from dotenv import load_dotenv
import json
import random
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms, models
from torchvision.io import read_image
from PIL import Image
import cv2

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("wildlife_classifier.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("wildlife_classifier")

# Load environment variables
load_dotenv()

# Set paths
DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
INATURALIST_DIR = DATA_DIR / "inaturalist"
MODELS_DIR = Path(os.getenv("MODELS_DIR", "./models"))
RESULTS_DIR = Path(os.getenv("RESULTS_DIR", "./results"))

# Check if GPU is available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_GPU = os.getenv("USE_GPU", "True").lower() == "true" and torch.cuda.is_available()

if USE_GPU:
    logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    logger.info("Using CPU")


class WildlifeDataset(Dataset):
    """Wildlife image dataset for PyTorch DataLoader."""
    
    def __init__(self, image_dir, annotations_file=None, transform=None, limit=None, generate_dummy=False):
        """
        Initialize the dataset.
        
        Args:
            image_dir (str): Directory containing images
            annotations_file (str, optional): Path to annotations file with labels
            transform (callable, optional): Optional transform to apply to images
            limit (int, optional): Limit number of samples (for testing purposes)
            generate_dummy (bool): Generate dummy data if real data isn't available
        """
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.images = []
        self.labels = []
        self.class_names = []
        
        # If we need to generate dummy data (for testing without the actual dataset)
        if generate_dummy:
            logger.info("Generating dummy wildlife dataset")
            self._generate_dummy_data(limit or 1000)
            return
            
        if not os.path.exists(image_dir):
            logger.warning(f"Image directory not found: {image_dir}")
            logger.info("Generating dummy wildlife dataset instead")
            self._generate_dummy_data(limit or 1000)
            return
        
        # If annotations file is provided, load real data
        if annotations_file and os.path.exists(annotations_file):
            self._load_from_annotations(annotations_file, limit)
        else:
            # Try to infer dataset from directory structure
            self._load_from_directory(limit)
    
    def _load_from_annotations(self, annotations_file, limit):
        """Load dataset from annotations file."""
        logger.info(f"Loading annotations from {annotations_file}")
        
        try:
            with open(annotations_file, 'r') as f:
                annotations = json.load(f)
            
            # Extract class names and create mapping
            if "categories" in annotations:
                self.class_names = [cat["name"] for cat in annotations["categories"]]
                class_id_to_idx = {cat["id"]: idx for idx, cat in enumerate(annotations["categories"])}
            else:
                # Assume annotations is a list of {image_path, class_id} entries
                unique_classes = sorted(list(set([ann["category_id"] for ann in annotations["annotations"]])))
                self.class_names = [f"Class_{c}" for c in unique_classes]
                class_id_to_idx = {c: idx for idx, c in enumerate(unique_classes)}
            
            # Extract image paths and labels
            for ann in annotations["annotations"]:
                if "file_name" in ann:
                    image_path = self.image_dir / ann["file_name"]
                else:
                    image_id = ann["image_id"]
                    # Find corresponding image path in images list
                    image_info = next((img for img in annotations["images"] if img["id"] == image_id), None)
                    if not image_info:
                        continue
                    image_path = self.image_dir / image_info["file_name"]
                
                if os.path.exists(image_path):
                    self.images.append(image_path)
                    self.labels.append(class_id_to_idx[ann["category_id"]])
                
                # Check if limit is reached
                if limit and len(self.images) >= limit:
                    break
            
            logger.info(f"Loaded {len(self.images)} images with {len(self.class_names)} classes")
            
        except Exception as e:
            logger.error(f"Error loading annotations: {e}")
            logger.info("Generating dummy wildlife dataset instead")
            self._generate_dummy_data(limit or 1000)
    
    def _load_from_directory(self, limit):
        """Infer dataset structure from directory (assumes subdirs are class names)."""
        logger.info(f"Loading dataset from directory structure: {self.image_dir}")
        
        try:
            # Assume subdirectories are class names
            subdirs = [d for d in os.listdir(self.image_dir) if os.path.isdir(os.path.join(self.image_dir, d))]
            
            if not subdirs:
                raise ValueError("No subdirectories found in image directory")
            
            self.class_names = subdirs
            
            # Collect image paths and labels
            for class_idx, class_name in enumerate(self.class_names):
                class_dir = os.path.join(self.image_dir, class_name)
                image_files = [f for f in os.listdir(class_dir) 
                              if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'))]
                
                for img_file in image_files:
                    self.images.append(os.path.join(class_dir, img_file))
                    self.labels.append(class_idx)
                    
                    # Check if limit is reached
                    if limit and len(self.images) >= limit:
                        break
                
                if limit and len(self.images) >= limit:
                    break
            
            logger.info(f"Loaded {len(self.images)} images with {len(self.class_names)} classes")
            
        except Exception as e:
            logger.error(f"Error loading from directory: {e}")
            logger.info("Generating dummy wildlife dataset instead")
            self._generate_dummy_data(limit or 1000)
    
    def _generate_dummy_data(self, num_samples):
        """Generate dummy data for testing."""
        logger.info(f"Generating {num_samples} dummy samples")
        
        # Create dummy class names (animal species)
        self.class_names = [
            "lion", "tiger", "bear", "elephant", "giraffe",
            "zebra", "gazelle", "deer", "wolf", "fox",
            "eagle", "owl", "parrot", "penguin", "flamingo"
        ]
        
        # Generate dummy data
        self.images = [f"dummy_image_{i}.jpg" for i in range(num_samples)]
        self.labels = [random.randint(0, len(self.class_names) - 1) for _ in range(num_samples)]
        
        logger.info(f"Generated {len(self.images)} dummy images with {len(self.class_names)} classes")
    
    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.images)
    
    def __getitem__(self, idx):
        """
        Return a sample from the dataset.
        
        Args:
            idx (int): Index of the sample
            
        Returns:
            tuple: (image, label)
        """
        # For dummy data, generate a random image
        if self.images[idx].startswith("dummy_"):
            # Create a random colored image
            img = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
            # Add some structure based on the class (to help the model learn something)
            label = self.labels[idx]
            color = [(0, 100, 0), (100, 0, 0), (0, 0, 100), (100, 100, 0), 
                    (100, 0, 100), (0, 100, 100), (100, 50, 0), (50, 100, 0),
                    (0, 50, 100), (100, 0, 50), (50, 0, 100), (0, 100, 50),
                    (70, 70, 0), (70, 0, 70), (0, 70, 70)][label % 15]
            
            # Draw a shape based on class
            shape_type = label % 3
            if shape_type == 0:  # Circle
                cv2.circle(img, (112, 112), 50, color, -1)
            elif shape_type == 1:  # Rectangle
                cv2.rectangle(img, (62, 62), (162, 162), color, -1)
            else:  # Triangle
                pts = np.array([[112, 62], [162, 162], [62, 162]], np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.fillPoly(img, [pts], color)
            
            # Convert to PIL Image
            image = Image.fromarray(img)
        else:
            # Load real image
            try:
                image = Image.open(self.images[idx]).convert("RGB")
            except Exception as e:
                logger.error(f"Error loading image {self.images[idx]}: {e}")
                # Return a blank image on error
                image = Image.new("RGB", (224, 224), (0, 0, 0))
        
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label


def get_transforms(train=True):
    """
    Get image transforms for training/testing.
    
    Args:
        train (bool): Whether to return training transforms (with augmentation)
        
    Returns:
        torchvision.transforms: Transform to apply to images
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(30),  # Increased rotation
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),  # Added affine transform
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Enhanced color jitter
            transforms.RandomGrayscale(p=0.05),  # Occasional grayscale
            transforms.ToTensor(),
            normalize,
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])


def create_model(num_classes, pretrained=True):
    """
    Create a pre-trained model with modified classifier head.
    
    Args:
        num_classes (int): Number of output classes
        pretrained (bool): Whether to use pretrained weights
        
    Returns:
        torch.nn.Module: Model
    """
    # Load a pre-trained model (ResNet50)
    model = models.resnet50(pretrained=pretrained)
    
    # Modify the final layer for our number of classes
    in_features = model.fc.in_features
    
    # Add dropout for regularization
    model.fc = nn.Sequential(
        nn.Dropout(0.5),  # Add dropout with 0.5 probability
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),  # Add another dropout layer
        nn.Linear(512, num_classes)
    )
    
    return model


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    """
    Train the model.
    
    Args:
        model (torch.nn.Module): Model to train
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        criterion: Loss function
        optimizer: Optimizer
        num_epochs (int): Number of training epochs
        
    Returns:
        tuple: (trained model, training history)
    """
    model = model.to(DEVICE)
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_acc = 0.0
    best_model_weights = None
    patience = 5  # Number of epochs to wait for improvement
    patience_counter = 0
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2
    )
    
    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch+1}/{num_epochs}")
        
        # Training phase
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        for inputs, labels in tqdm(train_loader, desc="Training"):
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc.item())
        
        logger.info(f"Train Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}")
        
        # Validation phase
        model.eval()
        val_running_loss = 0.0
        val_running_corrects = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Validation"):
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)
                
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                
                val_running_loss += loss.item() * inputs.size(0)
                val_running_corrects += torch.sum(preds == labels.data)
        
        val_epoch_loss = val_running_loss / len(val_loader.dataset)
        val_epoch_acc = val_running_corrects.double() / len(val_loader.dataset)
        
        history['val_loss'].append(val_epoch_loss)
        history['val_acc'].append(val_epoch_acc.item())
        
        logger.info(f"Val Loss: {val_epoch_loss:.4f}, Acc: {val_epoch_acc:.4f}")
        
        # Update learning rate scheduler
        scheduler.step(val_epoch_acc)
        
        # Save best model
        if val_epoch_acc > best_val_acc:
            best_val_acc = val_epoch_acc
            best_model_weights = model.state_dict().copy()
            logger.info(f"New best validation accuracy: {best_val_acc:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load best model weights
    model.load_state_dict(best_model_weights)
    logger.info(f"Best validation accuracy: {best_val_acc:.4f}")
    
    return model, history


def evaluate_model(model, test_loader, class_names):
    """
    Evaluate the model on test data.
    
    Args:
        model (torch.nn.Module): Trained model
        test_loader (DataLoader): Test data loader
        class_names (list): List of class names
        
    Returns:
        tuple: (accuracy, confusion matrix, classification report)
    """
    model = model.to(DEVICE)
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing"):
            inputs = inputs.to(DEVICE)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # Calculate accuracy
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    
    # Create confusion matrix and classification report
    from sklearn.metrics import confusion_matrix, classification_report
    cm = confusion_matrix(all_labels, all_preds)
    cr = classification_report(all_labels, all_preds, target_names=class_names)
    
    logger.info(f"Test Accuracy: {accuracy:.4f}")
    logger.info(f"Classification Report:\n{cr}")
    
    return accuracy, cm, cr


def visualize_predictions(model, dataset, class_names, num_images=5):
    """
    Visualize some predictions from the model.
    
    Args:
        model (torch.nn.Module): Trained model
        dataset (Dataset): Dataset to sample from
        class_names (list): List of class names
        num_images (int): Number of images to visualize
        
    Returns:
        matplotlib.figure.Figure: Figure with visualizations
    """
    model = model.to(DEVICE)
    model.eval()
    
    # Get a data loader with batch size 1
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    # Create a figure
    fig, axes = plt.subplots(num_images, 2, figsize=(12, 3 * num_images))
    
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloader):
            if i >= num_images:
                break
                
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            # Convert tensor to image
            img = inputs.cpu().squeeze().permute(1, 2, 0).numpy()
            img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img = np.clip(img, 0, 1)
            
            true_label = class_names[labels.item()]
            pred_label = class_names[preds.item()]
            
            # Plot original image
            axes[i, 0].imshow(img)
            axes[i, 0].set_title(f"True: {true_label}")
            axes[i, 0].axis("off")
            
            # Plot class probabilities
            probs = torch.nn.functional.softmax(outputs, dim=1).cpu().squeeze().numpy()
            
            # Sort by probability
            sorted_idx = np.argsort(probs)[::-1][:5]  # Top 5 predictions
            sorted_probs = probs[sorted_idx]
            sorted_names = [class_names[idx] for idx in sorted_idx]
            
            axes[i, 1].barh(range(len(sorted_names)), sorted_probs)
            axes[i, 1].set_yticks(range(len(sorted_names)))
            axes[i, 1].set_yticklabels(sorted_names)
            axes[i, 1].set_xlim(0, 1)
            axes[i, 1].set_xlabel("Probability")
            axes[i, 1].set_title("Predictions")
    
    plt.tight_layout()
    
    # Save figure
    fig_dir = RESULTS_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_dir / "wildlife_predictions.png")
    
    return fig


def save_model(model, class_names, save_path=None):
    """
    Save the trained model and class names.
    
    Args:
        model (torch.nn.Module): Trained model
        class_names (list): List of class names
        save_path (str, optional): Path to save the model. If None, use default path.
    """
    if save_path is None:
        model_dir = MODELS_DIR / "vision"
        model_dir.mkdir(parents=True, exist_ok=True)
        save_path = model_dir / "wildlife_classifier.pth"
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save model weights
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_names': class_names
    }, save_path)
    
    logger.info(f"Saved model to {save_path}")


def load_model(load_path=None, num_classes=None):
    """
    Load a trained model.
    
    Args:
        load_path (str, optional): Path to load the model from. If None, use default path.
        num_classes (int, optional): Number of classes. Required if loading just weights.
        
    Returns:
        tuple: (model, class_names)
    """
    if load_path is None:
        load_path = MODELS_DIR / "vision" / "wildlife_classifier.pth"
    
    try:
        checkpoint = torch.load(load_path, map_location=DEVICE)
        
        if 'class_names' in checkpoint:
            class_names = checkpoint['class_names']
            num_classes = len(class_names)
        elif num_classes is None:
            raise ValueError("Number of classes must be provided if not saved with model")
        else:
            class_names = [f"Class_{i}" for i in range(num_classes)]
        
        model = create_model(num_classes=num_classes, pretrained=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        logger.info(f"Loaded model from {load_path}")
        return model, class_names
    
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None, None


def plot_training_history(history):
    """
    Plot training history.
    
    Args:
        history (dict): Training history
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot loss
    axes[0].plot(history['train_loss'], label='Train')
    axes[0].plot(history['val_loss'], label='Validation')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    
    # Plot accuracy
    axes[1].plot(history['train_acc'], label='Train')
    axes[1].plot(history['val_acc'], label='Validation')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    
    plt.tight_layout()
    
    # Save figure
    fig_dir = RESULTS_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_dir / "wildlife_training_history.png")


def prepare_datasets(data_dir=None, batch_size=32, val_split=0.2, test_split=0.1, num_workers=4):
    """
    Prepare datasets and data loaders.
    
    Args:
        data_dir (str, optional): Directory containing images. If None, use default.
        batch_size (int): Batch size
        val_split (float): Validation split ratio
        test_split (float): Test split ratio
        num_workers (int): Number of worker processes for data loading
        
    Returns:
        tuple: (train_loader, val_loader, test_loader, class_names)
    """
    if data_dir is None:
        data_dir = INATURALIST_DIR
    
    # Check if directory exists
    if not os.path.exists(data_dir):
        logger.warning(f"Data directory not found: {data_dir}")
        logger.info("Using dummy data")
        generate_dummy = True
    else:
        generate_dummy = False
    
    # Get transforms
    train_transform = get_transforms(train=True)
    val_transform = get_transforms(train=False)
    
    # Create dataset
    full_dataset = WildlifeDataset(
        image_dir=data_dir,
        transform=val_transform,  # Use val transform initially (we'll set train transform later)
        generate_dummy=generate_dummy
    )
    
    # Get class names
    class_names = full_dataset.class_names
    
    # Split dataset
    dataset_size = len(full_dataset)
    val_size = int(val_split * dataset_size)
    test_size = int(test_split * dataset_size)
    train_size = dataset_size - val_size - test_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size]
    )
    
    # Apply different transforms to each split
    train_dataset = TransformDataset(train_dataset, train_transform)
    val_dataset = TransformDataset(val_dataset, val_transform)
    test_dataset = TransformDataset(test_dataset, val_transform)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    
    logger.info(f"Dataset split: {train_size} train, {val_size} validation, {test_size} test")
    
    return train_loader, val_loader, test_loader, class_names


class TransformDataset(Dataset):
    """Applies a transform to a dataset."""
    
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        if self.transform and not isinstance(image, torch.Tensor):
            image = self.transform(image)
        return image, label


def main():
    """Main function to train and evaluate the wildlife classification model."""
    logger.info("Starting wildlife classification pipeline")
    
    # Prepare datasets
    train_loader, val_loader, test_loader, class_names = prepare_datasets(
        batch_size=32, num_workers=2  # Adjust based on your hardware
    )
    
    # Create model
    model = create_model(num_classes=len(class_names), pretrained=True)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), 
        lr=0.001, 
        weight_decay=1e-4  # Add L2 regularization
    )
    
    # Train model
    trained_model, history = train_model(
        model, 
        train_loader, 
        val_loader, 
        criterion, 
        optimizer,
        num_epochs=10  # Increased number of epochs
    )
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate model
    accuracy, confusion_matrix, report = evaluate_model(
        trained_model, test_loader, class_names
    )
    
    # Visualize predictions
    visualize_predictions(trained_model, test_loader.dataset, class_names)
    
    # Save model
    save_model(trained_model, class_names)
    
    logger.info("Wildlife classification pipeline completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main()) 