#!/usr/bin/env python3
"""
Training script for Suspicious Activity Detection using Pose Features.

This script:
1. Loads pre-extracted pose features
2. Applies data augmentation (keypoint dropout)
3. Trains a SuspiciousActivityLSTM model
4. Evaluates on validation set
5. Saves the best model
"""

import os
import json
import time
import pickle
import argparse
import random
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import shutil
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# Visualization functions
def plot_training_curves(train_losses, val_losses, train_accs, val_accs, output_dir):
    """
    Plot training and validation curves.
    
    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        train_accs: List of training accuracies per epoch
        val_accs: List of validation accuracies per epoch
        output_dir: Directory to save the plot
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot loss curves
        epochs = range(1, len(train_losses) + 1)
        ax1.plot(epochs, train_losses, 'b-', label='Train Loss')
        ax1.plot(epochs, val_losses, 'r-', label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Plot accuracy curves
        ax2.plot(epochs, train_accs, 'b-', label='Train Accuracy')
        ax2.plot(epochs, val_accs, 'r-', label='Validation Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        # Adjust layout and save
        plt.tight_layout()
        output_path = os.path.join(output_dir, 'training_curves.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved training curves to {output_path}")
        
    except Exception as e:
        print(f"Error plotting training curves: {str(e)}")
        if 'plt' in locals():
            plt.close()

def plot_confusion_matrix(all_labels, all_probs, output_dir):
    """
    Plot and save confusion matrix.
    
    Args:
        all_labels: List of true labels (0 or 1)
        all_probs: List of predicted probabilities
        output_dir: Directory to save the plot
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert probabilities to binary predictions
        preds = (np.array(all_probs) > 0.5).astype(int)
        
        # Calculate confusion matrix
        cm = confusion_matrix(all_labels, preds)
        
        # Create figure
        plt.figure(figsize=(8, 6))
        
        # Plot confusion matrix
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            cbar=False,
            xticklabels=['Non-Fight', 'Fight'],
            yticklabels=['Non-Fight', 'Fight'],
            annot_kws={"size": 14}
        )
        
        # Add labels and title
        plt.title('Confusion Matrix', fontsize=14, pad=20)
        plt.ylabel('True Label', fontsize=12, labelpad=10)
        plt.xlabel('Predicted Label', fontsize=12, labelpad=10)
        
        # Adjust layout and save
        plt.tight_layout()
        output_path = os.path.join(output_dir, 'confusion_matrix.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved confusion matrix to {output_path}")
        
    except Exception as e:
        print(f"Error plotting confusion matrix: {str(e)}")
        if 'plt' in locals():
            plt.close()

def plot_roc_curve(all_labels, all_probs, output_dir):
    """
    Plot and save ROC curve.
    
    Args:
        all_labels: List of true labels (0 or 1)
        all_probs: List of predicted probabilities
        output_dir: Directory to save the plot
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Calculate ROC curve and AUC
        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        roc_auc = auc(fpr, tpr)
        
        # Create figure
        plt.figure(figsize=(8, 6))
        
        # Plot ROC curve
        plt.plot(
            fpr, 
            tpr, 
            color='#1f77b4', 
            lw=2, 
            label=f'ROC curve (AUC = {roc_auc:.3f})'
        )
        
        # Plot random classifier line
        plt.plot(
            [0, 1], 
            [0, 1], 
            color='navy', 
            lw=1, 
            linestyle='--',
            label='Random (AUC = 0.500)'
        )
        
        # Customize plot
        plt.xlim([-0.01, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12, labelpad=10)
        plt.ylabel('True Positive Rate', fontsize=12, labelpad=10)
        plt.title('Receiver Operating Characteristic', fontsize=14, pad=20)
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Adjust layout and save
        plt.tight_layout()
        output_path = os.path.join(output_dir, 'roc_curve.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved ROC curve to {output_path}")
        
        return roc_auc
        
    except Exception as e:
        print(f"Error plotting ROC curve: {str(e)}")
        if 'plt' in locals():
            plt.close()
        return 0.0

# Import model
from models import SuspiciousActivityLSTM, count_parameters


class TrainingConfig:
    """Configuration class for training parameters."""
    
    def __init__(self):
        # Data parameters
        self.features_dir = 'data/features_smoothed'
        self.output_dir = f'runs/exp_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        
        # Directory structure (will be set in setup_directories)
        self.checkpoint_dir = ''
        self.figures_dir = ''
        self.logs_dir = ''
        self.metrics_file = ''
        self.final_metrics_file = ''
        
        # Model architecture
        self.sequence_length = 30
        self.hidden_size = 256
        self.num_layers = 2
        self.dropout = 0.5
        self.bidirectional = True
        self.use_attention = True
        self.num_attention_heads = 4
        
        # Training parameters
        self.batch_size = 32
        self.num_epochs = 100
        self.learning_rate = 3e-4
        self.weight_decay = 1e-4  # L2 regularization
        self.patience = 15        # For early stopping
        self.checkpoint_freq = 5  # Save checkpoint every N epochs
        self.num_workers = 4      # For data loading
        
        # Device configuration
        self.device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def setup_directories(self):
        """Create necessary directories for the experiment."""
        # Create main output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Define directory structure
        self.checkpoint_dir = os.path.join(self.output_dir, 'checkpoints')
        self.figures_dir = os.path.join(self.output_dir, 'figures')
        self.logs_dir = os.path.join(self.output_dir, 'logs')
        self.metrics_file = os.path.join(self.output_dir, 'metrics.json')
        self.final_metrics_file = os.path.join(self.output_dir, 'final_metrics.json')
        
        # Create all directories
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.figures_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        
        # Save the config
        self.save(os.path.join(self.output_dir, 'config.json'))
    
    def save(self, filepath: str = None):
        """Save configuration to a JSON file."""
        if filepath is None:
            filepath = os.path.join(self.output_dir, 'config.json')
            
        # Don't save directory paths to config
        skip_keys = {'checkpoint_dir', 'figures_dir', 'logs_dir', 
                    'metrics_file', 'final_metrics_file'}
        
        config_dict = {k: v for k, v in self.__dict__.items() 
                      if k not in skip_keys}
        
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)
    
    @classmethod
    def load(cls, filepath: str) -> 'TrainingConfig':
        """Load configuration from a JSON file."""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        config = cls()
        for key, value in config_dict.items():
            setattr(config, key, value)
        
        # Recreate directory structure
        config.setup_directories()
        return config

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Constants
NUM_JOINTS = 17  # COCO keypoints
NUM_PEOPLE = 2
FEATURES_PER_PERSON = 102
TOTAL_FEATURES = NUM_PEOPLE * FEATURES_PER_PERSON

# Keypoint groups for dropout (COCO keypoint indices)
KEYPOINT_GROUPS = {
    'left_arm': [5, 7, 9],      # shoulder, elbow, wrist
    'right_arm': [6, 8, 10],    # shoulder, elbow, wrist
    'left_leg': [11, 13, 15],   # hip, knee, ankle
    'right_leg': [12, 14, 16],  # hip, knee, ankle
    'torso': [5, 6, 11, 12],    # shoulders and hips
    'head': [0, 1, 2, 3, 4],    # nose, eyes, ears
}

class PoseSequenceDataset(Dataset):
    """Dataset for loading pose sequences with keypoint dropout augmentation."""
    
    def __init__(self, data_dir: str, split: str = 'train', augment: bool = False, dropout_prob: float = 0.3):
        """
        Initialize the dataset.
        
        Args:
            data_dir: Root directory containing train/val/test subdirectories
            split: One of 'train', 'val', or 'test'
            augment: Whether to apply data augmentation (keypoint dropout)
            dropout_prob: Probability of applying keypoint dropout (only used if augment=True)
        """
        self.data_dir = os.path.join(data_dir, split)
        self.augment = augment and (split == 'train')  # Only augment training data
        self.dropout_prob = dropout_prob
        
        # Collect all feature files
        self.samples = []
        self.class_names = ['Fight', 'NonFight']  # Match exact directory names
        
        for label in self.class_names:
            label_dir = os.path.join(self.data_dir, label)
            if not os.path.exists(label_dir):
                print(f"Warning: Directory not found: {label_dir}")
                continue
                
            label_idx = 1 if label == 'Fight' else 0
            npy_files = [f for f in os.listdir(label_dir) if f.endswith('.npy')]
            
            if not npy_files:
                print(f"Warning: No .npy files found in {label_dir}")
                continue
                
            for fname in npy_files:
                self.samples.append((
                    os.path.join(label_dir, fname),
                    label_idx
                ))
                
            print(f"Loaded {len(npy_files)} samples from {label_dir}")
        
        print(f"Loaded {len(self.samples)} samples from {self.data_dir}")
    
    def __len__(self) -> int:
        return len(self.samples)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            tuple: (sequence, label) where sequence is a float32 tensor of shape [seq_len, features]
                  and label is 0 for NonFight or 1 for Fight
        """
        file_path, label = self.samples[idx]
        
        try:
            # Load the sequence
            sequence = np.load(file_path).astype(np.float32)  # [seq_len, features]
            
            # Apply keypoint dropout augmentation if training
            if self.augment and random.random() < self.dropout_prob:
                sequence = self.keypoint_dropout(sequence)
            
            # Convert to tensor
            sequence = torch.from_numpy(sequence)  # [seq_len, features]
            
            return sequence, label
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            # Return a zero tensor with the expected shape if there's an error
            return torch.zeros(1, 102), torch.tensor(-1, dtype=torch.long)
    
    def keypoint_dropout(self, sequence_features: np.ndarray) -> np.ndarray:
        """
        Applies keypoint dropout by zeroing out features for a random body part.
        This version correctly handles the interleaved feature structure.
        
        Args:
            sequence_features: Input sequence of shape [seq_len, features]
            
        Returns:
            Augmented sequence with some keypoints dropped
        """
        if not self.augment or np.random.rand() >= self.dropout_prob:
            return sequence_features
            
        # Clone to avoid modifying the original array
        sequence_features = sequence_features.copy()
        
        # Body parts defined by keypoint indices (0-16)
        body_parts = {
            "left_arm": [5, 7, 9],      # LShoulder, LElbow, LWrist
            "right_arm": [6, 8, 10],    # RShoulder, RElbow, RWrist
            "left_leg": [11, 13, 15],   # LHip, LKnee, LAnkle
            "right_leg": [12, 14, 16],  # RHip, RKnee, RAnkle
            "torso": [5, 6, 11, 12]     # Shoulders and Hips
        }
        
        # Select a random body part to drop
        part_name = np.random.choice(list(body_parts.keys()))
        keypoint_indices_to_drop = body_parts[part_name]
        
        # Define feature block sizes
        num_kps = 17
        coords_per_kp = 2  # (x, y)
        features_per_person = num_kps * coords_per_kp * 3  # (pos, vel, acc) = 102
        
        # Zero out features for each selected keypoint
        for kp_idx in keypoint_indices_to_drop:
            for person_idx in range(2):  # For person 1 and person 2
                person_offset = person_idx * features_per_person
                
                # Zero out relative position features (x, y)
                pos_start_idx = person_offset + kp_idx * coords_per_kp
                sequence_features[:, pos_start_idx:pos_start_idx + coords_per_kp] = 0
                
                # Zero out velocity features (vx, vy)
                vel_start_idx = person_offset + (num_kps * coords_per_kp) + (kp_idx * coords_per_kp)
                sequence_features[:, vel_start_idx:vel_start_idx + coords_per_kp] = 0
                
                # Zero out acceleration features (ax, ay)
                acc_start_idx = person_offset + (num_kps * coords_per_kp * 2) + (kp_idx * coords_per_kp)
                sequence_features[:, acc_start_idx:acc_start_idx + coords_per_kp] = 0

        return sequence_features
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            tuple: (sequence, label) where sequence is a float32 tensor of shape [seq_len, features]
        """
        file_path, label = self.samples[idx]
        
        # Load the sequence
        sequence = np.load(file_path).astype(np.float32)  # [seq_len, features]
        
        # Apply keypoint dropout augmentation
        if self.augment:
            sequence = self.keypoint_dropout(sequence)
        
        # Convert to tensor
        sequence = torch.from_numpy(sequence)  # [seq_len, features]
        
        return sequence, label


def create_dataloaders(
    data_dir: str, 
    batch_size: int = 32,
    num_workers: int = 4,
    dropout_prob: float = 0.3
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create data loaders for train, validation, and test sets.
    
    Args:
        data_dir: Root directory containing train/val/test subdirectories
        batch_size: Batch size
        num_workers: Number of worker processes for data loading
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create datasets with consistent dropout_prob for all splits
    # (though only train will use it for augmentation)
    train_dataset = PoseSequenceDataset(
        data_dir, 
        split='train', 
        augment=True,
        dropout_prob=dropout_prob
    )
    val_dataset = PoseSequenceDataset(
        data_dir, 
        split='val', 
        augment=False,
        dropout_prob=dropout_prob  # Pass it for consistency, though augment=False
    )
    test_dataset = PoseSequenceDataset(
        data_dir, 
        split='test', 
        augment=False,
        dropout_prob=dropout_prob  # Pass it for consistency, though augment=False
    )
    
    # Validate that no dataset is empty
    if len(train_dataset) == 0 or len(val_dataset) == 0 or len(test_dataset) == 0:
        raise ValueError(
            "One or more datasets are empty. "
            f"Train samples: {len(train_dataset)}, "
            f"Val samples: {len(val_dataset)}, "
            f"Test samples: {len(test_dataset)}. "
            "Please check that the data directory contains the expected files and class names match."
        )
    
    print(f"Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Standard shuffling for balanced dataset
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True  # Drop last incomplete batch
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    writer: Optional[SummaryWriter] = None
) -> float:
    """
    Train the model for one epoch.
    
    Args:
        model: The model to train
        train_loader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer
        device: Device to run on
        epoch: Current epoch number
        writer: TensorBoard writer
        
    Returns:
        Average training loss for the epoch
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]", leave=False)
    
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs = inputs.to(device)
        targets = targets.float().to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass - get logits for the positive class (class 1)
        outputs = model(inputs)
        # For binary classification, we can take the logits for class 1
        # and use them with BCEWithLogitsLoss
        outputs = outputs[:, 1]  # Get the logits for class 1
        loss = criterion(outputs, targets)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        preds = torch.sigmoid(outputs) > 0.5
        correct += preds.eq(targets).sum().item()
        total += targets.size(0)
        
        # Update progress bar
        pbar.set_postfix({
            'loss': running_loss / (batch_idx + 1),
            'acc': 100. * correct / total
        })
    
    # Log to TensorBoard
    avg_loss = running_loss / len(train_loader)
    accuracy = 100. * correct / total
    
    if writer is not None:
        writer.add_scalar('Loss/train', avg_loss, epoch)
        writer.add_scalar('Accuracy/train', accuracy, epoch)
    
    return avg_loss, accuracy


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    writer: Optional[SummaryWriter] = None
) -> Tuple[float, float]:
    """
    Validate the model.
    
    Args:
        model: The model to validate
        val_loader: DataLoader for validation data
        criterion: Loss function
        device: Device to run on
        epoch: Current epoch number
        writer: TensorBoard writer
        
    Returns:
        Tuple of (average loss, accuracy)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    all_outputs = []
    all_targets = []
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]", leave=False)
        
        for inputs, targets in pbar:
            inputs = inputs.to(device)
            targets = targets.float().to(device)
            
            # Forward pass - get logits for the positive class (class 1)
            outputs = model(inputs)
            # For binary classification, we take the logits for class 1
            outputs = outputs[:, 1]  # Get the logits for class 1
            loss = criterion(outputs, targets)
            
            # Statistics
            running_loss += loss.item()
            preds = torch.sigmoid(outputs) > 0.5
            correct += preds.eq(targets).sum().item()
            total += targets.size(0)
            
            # Store for metrics
            all_outputs.extend(torch.sigmoid(outputs).cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({
                'loss': running_loss / (pbar.n + 1),
                'acc': 100. * correct / total
            })
    
    # Calculate metrics
    avg_loss = running_loss / len(val_loader)
    accuracy = 100. * correct / total
    
    # Log to TensorBoard
    if writer is not None:
        writer.add_scalar('Loss/val', avg_loss, epoch)
        writer.add_scalar('Accuracy/val', accuracy, epoch)
        
        # Log histograms of predictions
        writer.add_histogram('Predictions/val', np.array(all_outputs), epoch)
    
    return avg_loss, accuracy


def save_checkpoint(
    state: Dict,
    is_best: bool,
    filename: str = 'checkpoint.pth.tar',
    best_filename: str = 'model_best.pth.tar'
) -> None:
    """Save checkpoint and best model."""
    torch.save(state, filename)
    if is_best:
        torch.save(state, best_filename)


def train(
    data_dir: str,
    model_dir: str = None,  # Will be set by config
    batch_size: int = 32,
    num_epochs: int = 50,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
    dropout: float = 0.5,
    num_workers: int = 4,
    device: str = None,
    config: TrainingConfig = None
) -> Dict[str, float]:
    """
    Main training function.
    
    Args:
        data_dir: Directory containing train/val/test subdirectories with .npy files
        model_dir: Directory to save model checkpoints and logs
        batch_size: Batch size for training
        num_epochs: Number of epochs to train for
        learning_rate: Initial learning rate
        weight_decay: Weight decay for Adam optimizer
        num_workers: Number of worker processes for data loading
        device: Device to run on ('cuda', 'mps', or 'cpu')
    """
    # Initialize config if not provided
    if config is None:
        config = TrainingConfig()
        config.batch_size = batch_size
        config.num_epochs = num_epochs
        config.learning_rate = learning_rate
        config.weight_decay = weight_decay
        config.dropout = dropout
        config.num_workers = num_workers
        config.device = device or config.device
        
        # Set model directory if not specified
        if model_dir:
            config.output_dir = model_dir
    else:
        # Update config with provided arguments
        if model_dir:
            config.output_dir = model_dir
        if batch_size:
            config.batch_size = batch_size
        if num_epochs:
            config.num_epochs = num_epochs
        if learning_rate:
            config.learning_rate = learning_rate
        if weight_decay:
            config.weight_decay = weight_decay
        if dropout is not None:
            config.dropout = dropout
        if num_workers:
            config.num_workers = num_workers
        if device:
            config.device = device
    
    # Set up directory structure
    config.setup_directories()
    print(f"Saving outputs to: {os.path.abspath(config.output_dir)}")
    
    # Set device
    device = torch.device(config.device)
    
    # Save configuration
    config.save()
    
    # Set up logging
    writer = SummaryWriter(log_dir=config.logs_dir)
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=config.features_dir,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        dropout_prob=config.dropout
    )
    
    # Initialize model with configurable dropout
    print(f"Initializing model with dropout={config.dropout}...")
    input_size = train_loader.dataset[0][0].shape[1]  # Feature size
    model = SuspiciousActivityLSTM(
        input_size=input_size,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        dropout=config.dropout,
        bidirectional=config.bidirectional
    ).to(device)
    
    print(f"Model parameters: {count_parameters(model):,}")
    print(f"Input size: {input_size}")
    
    # Loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    
    # Training loop
    print("Starting training...")
    best_acc = 0.0
    best_epoch = 0
    early_stop_counter = 0
    best_val_acc = 0.0  # Initialize best validation accuracy
    
    # For storing metrics
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    all_metrics = {
        'train_losses': [],
        'val_losses': [],
        'train_accs': [],
        'val_accs': [],
        'best_epoch': 0,
        'best_val_acc': 0.0,
        'config': config.__dict__
    }
    
    # For final evaluation
    all_labels = []
    all_probs = []
    all_preds = []
    
    for epoch in range(config.num_epochs):
        # Train for one epoch
        train_loss, train_acc = train_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            writer=writer
        )
        
        # Evaluate on validation set
        if len(val_loader) > 0:
            val_loss, val_acc = validate(
                model, val_loader, criterion, device, epoch, writer
            )
            # Update learning rate based on validation accuracy
            scheduler.step(val_acc)
        else:
            val_loss, val_acc = 0.0, 0.0
            print("Warning: Validation set is empty, skipping validation and learning rate scheduling")
        
        # Store metrics
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Handle validation metrics
        if len(val_loader) > 0:
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            
            # Save checkpoint based on validation accuracy
            is_best = val_acc > best_val_acc
            if is_best:
                best_val_acc = val_acc
                best_epoch = epoch
        else:
            # If no validation set, save checkpoint based on training accuracy
            is_best = train_acc > best_val_acc
            if is_best:
                best_val_acc = train_acc
                best_epoch = epoch
        
        if is_best:
            # Save best model
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_val_acc': best_val_acc,
                'optimizer': optimizer.state_dict(),
                'config': config
            }, is_best, 
            filename=os.path.join(config.checkpoint_dir, f'checkpoint_epoch{epoch+1}.pth.tar'),
            best_filename=os.path.join(config.checkpoint_dir, 'model_best.pth.tar')
            )
            
            # Save metrics
            metrics = {
                'train_loss': train_loss,
                'train_acc': train_acc,
                'epoch': epoch + 1,
                'best_val_acc': best_acc,
                'best_epoch': best_epoch
            }
            
            # Add validation metrics if available
            if len(val_loader) > 0:
                metrics.update({
                    'val_loss': val_loss,
                    'val_acc': val_acc
                })
            
            with open(config.metrics_file, 'w') as f:
                json.dump(metrics, f, indent=2)
        
        # Save checkpoint periodically
        if (epoch + 1) % config.checkpoint_freq == 0:
            checkpoint_path = os.path.join(
                config.checkpoint_dir, 
                f'checkpoint_epoch_{epoch+1}.pth.tar'
            )
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'config': {k: v for k, v in config.__dict__.items() 
                         if k not in {'checkpoint_dir', 'figures_dir', 'logs_dir',
                                    'metrics_file', 'final_metrics_file'}},
                'val_acc': float(val_acc),
                'train_acc': float(train_acc)
            }, checkpoint_path)
        
        # Save metrics after each epoch
        final_metrics = {
            'train_losses': [float(loss) for loss in train_losses],
            'val_losses': [float(loss) for loss in val_losses],
            'train_accs': [float(acc) for acc in train_accs],
            'val_accs': [float(acc) for acc in val_accs],
            'best_epoch': int(best_epoch),
            'best_val_acc': float(best_val_acc),
            'current_epoch': int(epoch + 1)
        }
        
        with open(os.path.join(config.output_dir, 'metrics.json'), 'w') as f:
            json.dump(final_metrics, f, indent=2)
        
        # Print progress
        print(f'Epoch {epoch+1}/{config.num_epochs} - '
              f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
              f'Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%, '
              f'Best Val Acc: {best_val_acc:.2f}% @ Epoch {best_epoch+1}')
        
        # Early stopping
        if early_stop_counter >= config.patience:
            print(f'Early stopping triggered after {epoch+1} epochs')
            break
    
    # Load the best model for final evaluation
    best_model_path = os.path.join(config.checkpoint_dir, 'model_best.pth.tar')
    
    # First try with weights_only=True (default in PyTorch 2.6+)
    try:
        # Add TrainingConfig to safe globals if it exists
        if 'TrainingConfig' in globals():
            torch.serialization.add_safe_globals([globals()['TrainingConfig']])
            
        best_checkpoint = torch.load(
            best_model_path,
            weights_only=True,
            map_location=config.device
        )
    except (pickle.UnpicklingError, RuntimeError, TypeError) as e:
        print(f"Warning: Could not load with weights_only=True, falling back to weights_only=False\n{e}")
        try:
            # Fall back to weights_only=False with custom object loading
            with open(best_model_path, 'rb') as f:
                best_checkpoint = torch.load(
                    f,
                    weights_only=False,
                    map_location=config.device,
                    pickle_module=pickle
                )
        except Exception as e:
            print(f"Error loading model: {e}")
            # Try one more time with the most permissive settings
            try:
                with open(best_model_path, 'rb') as f:
                    best_checkpoint = torch.load(
                        f,
                        map_location=config.device,
                        pickle_module=pickle
                    )
            except Exception as e2:
                print(f"Fatal error loading model: {e2}")
                # Return the model as is if we can't load the checkpoint
                return model
    
    # Load model state dict
    if 'state_dict' in best_checkpoint:
        model.load_state_dict(best_checkpoint['state_dict'])
        print(f"Loaded best model from epoch {best_checkpoint.get('epoch', 'N/A')} "
              f"with val acc {best_checkpoint.get('best_val_accuracy', 0.0):.2f}%")
    else:
        # Handle case where the checkpoint is just the state dict
        model.load_state_dict(best_checkpoint)
        print("Loaded model state dict")
    
    # Final evaluation on test set
    test_loss, test_accuracy = validate(model, test_loader, criterion, device, epoch, writer)
    print(f'\nTest Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')
    
    # Save final model with proper serialization
    final_model_path = os.path.join(config.checkpoint_dir, 'model_final.pth.tar')
    checkpoint = {
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'config': {k: v for k, v in config.__dict__.items() 
                  if k not in {'checkpoint_dir', 'figures_dir', 'logs_dir',
                              'metrics_file', 'final_metrics_file'}},
        'test_accuracy': test_accuracy,
        'test_loss': test_loss,
        'best_val_accuracy': best_val_acc,
        'best_epoch': best_epoch,
        'model_class': model.__class__.__name__,
        'model_config': getattr(model, 'config', {})
    }
    
    # Use a more robust saving method
    try:
        # First try with the standard save
        torch.save(checkpoint, final_model_path, _use_new_zipfile_serialization=True)
        
        # Verify the saved model can be loaded
        try:
            test_load = torch.load(final_model_path, map_location='cpu', weights_only=True)
            if 'state_dict' not in test_load:
                raise ValueError("Saved model verification failed: Invalid checkpoint format")
        except Exception as e:
            print(f"Warning: Saved model verification failed, retrying with different format: {e}")
            # If verification fails, try with a different format
            torch.save(checkpoint, final_model_path, _use_new_zipfile_serialization=False)
    except Exception as e:
        print(f"Error saving model: {e}")
        raise
    
    # Generate final predictions for metrics
    model.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(config.device)
            labels = labels.to(config.device)
            
            outputs = model(inputs)
            probs = torch.sigmoid(outputs).squeeze()
            preds = (probs > 0.5).long()
            
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    
    # Save final metrics
    metrics = {
        'test_accuracy': test_accuracy,
        'test_loss': test_loss,
        'best_val_accuracy': best_val_acc,
        'num_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs
    }
    
    # Plot training curves
    plot_training_curves(train_losses, val_losses, train_accs, val_accs, config.figures_dir)
    
    # Save metrics
    with open(config.metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Plot confusion matrix and ROC curve if we have the data
    if 'all_labels' in locals() and 'all_probs' in locals():
        # Ensure we have numpy arrays and handle different input formats
        all_labels_np = np.array(all_labels)
        all_probs_np = np.array(all_probs)
        
        # Convert to 1D arrays if needed
        if len(all_labels_np.shape) > 1 and all_labels_np.shape[1] > 1:
            # If one-hot encoded, convert to 1D array of class indices
            all_labels_np = np.argmax(all_labels_np, axis=1)
        all_labels_np = all_labels_np.astype(int)  # Ensure integer type for labels
        
        # Convert probabilities to 1D array
        if len(all_probs_np.shape) > 1 and all_probs_np.shape[1] > 1:
            # If probabilities for both classes, take probability of positive class
            all_probs_np = all_probs_np[:, 1]
        
        # Plot confusion matrix with properly formatted inputs
        plot_confusion_matrix(all_labels_np, all_probs_np, config.figures_dir)
        
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(all_labels_np, all_probs_np)
        roc_auc = auc(fpr, tpr)
        
        # Plot ROC curve
        plot_roc_curve(all_labels_np, all_probs_np, config.figures_dir)
        
        # Convert predictions to binary (0 or 1) based on threshold
        # Convert predictions to binary (0 or 1) based on threshold
        if isinstance(all_probs, np.ndarray):
            if all_probs.ndim > 1 and all_probs.shape[1] > 1:
                # If all_probs is 2D with multiple columns, take the second column (assuming binary classification)
                y_pred = (all_probs[:, 1] > 0.5).astype(int)
            else:
                # If all_probs is 1D or 2D with single column
                y_pred = (all_probs.reshape(-1) > 0.5).astype(int)
        else:
            # If all_probs is a list or other sequence
            y_pred = (np.array(all_probs) > 0.5).astype(int)
            
        # Ensure labels are 1D array of integers (0 or 1)
        if isinstance(all_labels_np, np.ndarray):
            if all_labels_np.ndim > 1 and all_labels_np.shape[1] > 1:
                # If one-hot encoded, convert to binary
                y_true = np.argmax(all_labels_np, axis=1).astype(int)
            else:
                # If already binary, flatten and convert to int
                y_true = all_labels_np.reshape(-1).astype(int)
        else:
            # If not a numpy array, convert to numpy first
            y_true = np.array(all_labels_np).reshape(-1).astype(int)
            
        # Helper function to convert numpy types to Python native types
        def convert_to_python(obj):
            if isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_python(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_python(x) for x in obj]
            return obj
        
        # Calculate metrics with consistent formats
        try:
            # Ensure y_pred is 1D for confusion matrix
            if y_pred.ndim > 1 and y_pred.shape[1] > 1:
                y_pred_labels = np.argmax(y_pred, axis=1)
            else:
                y_pred_labels = y_pred.reshape(-1)
                
            cm = confusion_matrix(y_true, y_pred_labels)
            
            # Convert all numpy arrays to Python native types
            test_metrics = {
                'test_accuracy': float(test_accuracy),
                'test_loss': float(test_loss),
                'best_val_accuracy': float(best_val_acc),
                'roc_auc': float(roc_auc) if roc_auc is not None else None,
                'confusion_matrix': cm.tolist(),
                'roc_curve': {
                    'fpr': fpr.tolist() if fpr is not None else [],
                    'tpr': tpr.tolist() if tpr is not None else [],
                    'thresholds': _.tolist() if _ is not None else []
                },
                'y_true': y_true.tolist(),
                'y_pred': y_pred_labels.tolist()
            }
            
            # Ensure all values are JSON serializable
            test_metrics = convert_to_python(test_metrics)
            
        except Exception as e:
            print(f"Error calculating metrics: {e}")
            print(f"y_true shape: {np.array(y_true).shape}, type: {type(y_true[0]) if len(y_true) > 0 else 'empty'}")
            print(f"y_pred shape: {np.array(y_pred).shape}, type: {type(y_pred[0]) if len(y_pred) > 0 else 'empty'}")
            # Fallback to minimal metrics
            test_metrics = {
                'test_accuracy': float(test_accuracy),
                'test_loss': float(test_loss),
                'best_val_accuracy': float(best_val_acc),
                'error': str(e)
            }
        
        with open(config.final_metrics_file, 'w') as f:
            json.dump(test_metrics, f, indent=2)
    
    # Close TensorBoard writer
    writer.close()
    
    # Verify output structure
    verify_output_structure(config)
    
    return metrics


def verify_output_structure(config) -> None:
    """Verify that the output directory structure is correct and all necessary files exist.
    
    Args:
        config: TrainingConfig instance with directory paths
    """
    output_dir = config.output_dir
    
    # Required directories
    required_dirs = [
        output_dir,
        config.checkpoint_dir,
        config.figures_dir,
        config.logs_dir
    ]
    
    # Required files
    required_files = [
        os.path.join(output_dir, 'config.json'),
        config.metrics_file,
        config.final_metrics_file,
        os.path.join(config.checkpoint_dir, 'model_best.pth.tar'),
        os.path.join(config.checkpoint_dir, 'model_final.pth.tar'),
        os.path.join(config.figures_dir, 'training_curves.png'),
        os.path.join(config.figures_dir, 'confusion_matrix.png'),
        os.path.join(config.figures_dir, 'roc_curve.png')
    ]
    
    print("\n" + "="*70)
    print(f"VERIFYING OUTPUT DIRECTORY: {os.path.abspath(output_dir)}")
    print("="*70)
    
    # Check directories
    print("\nREQUIRED DIRECTORIES:")
    dir_status = []
    for dir_path in required_dirs:
        exists = os.path.exists(dir_path)
        dir_status.append((dir_path, exists))
        print(f"  {'✓' if exists else '✗'} {os.path.relpath(dir_path, output_dir) or '.'}")
    
    # Check files
    print("\nREQUIRED FILES:")
    file_status = []
    for file_path in required_files:
        exists = os.path.exists(file_path)
        size = os.path.getsize(file_path) if exists else 0
        file_status.append((file_path, exists, size))
        print(f"  {'✓' if exists else '✗'} {os.path.relpath(file_path, output_dir)} "
              f"({size/1024:.1f} KB)")
    
    # Check TensorBoard logs
    print("\nTENSORBOARD LOGS:")
    if os.path.exists(config.logs_dir):
        log_files = [f for f in os.listdir(config.logs_dir) 
                    if f.startswith('events.out.tfevents')]
        if log_files:
            print(f"  ✓ Found {len(log_files)} TensorBoard log file(s)")
            for log_file in sorted(log_files)[-2:]:  # Show most recent 2 logs
                size = os.path.getsize(os.path.join(config.logs_dir, log_file)) / 1024
                print(f"     • {log_file} ({size:.1f} KB)")
        else:
            print("  ✗ No TensorBoard log files found")
    else:
        print(f"  ✗ Logs directory does not exist: {config.logs_dir}")
    
    # Summary
    print("\n" + "="*70)
    print("VERIFICATION SUMMARY:")
    print(f"  ✓ Directories: {sum(1 for _, exists in dir_status if exists)}/{len(dir_status)} found")
    print(f"  ✓ Files: {sum(1 for _, exists, _ in file_status if exists)}/{len(file_status)} found")
    
    # Check for critical issues
    critical_issues = []
    for path, exists in dir_status:
        if not exists:
            critical_issues.append(f"Missing directory: {os.path.relpath(path, output_dir) or '.'}")
    
    for path, exists, size in file_status:
        if not exists:
            critical_issues.append(f"Missing file: {os.path.relpath(path, output_dir)}")
        elif size == 0:
            critical_issues.append(f"Empty file: {os.path.relpath(path, output_dir)}")
    
    if critical_issues:
        print("\nCRITICAL ISSUES FOUND:")
        for issue in critical_issues:
            print(f"  ✗ {issue}")
    else:
        print("\n✓ All required files and directories are present")
    
    print("="*70 + "\n")
    
    return not bool(critical_issues)


def main():
    parser = argparse.ArgumentParser(description='Train Suspicious Activity Detection Model')
    parser.add_argument('--data-dir', type=str, default='data/features_smoothed',
                        help='Directory containing train/val/test subdirectories with .npy files')
    parser.add_argument('--model-dir', type=str, default='runs/exp',
                        help='Directory to save model checkpoints and logs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs to train for')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='Weight decay for Adam optimizer')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout probability')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of worker processes for data loading')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to run on (cuda, mps, or cpu)')
    parser.add_argument('--hidden-size', type=int, default=256,
                        help='Hidden size for LSTM layers')
    parser.add_argument('--num-layers', type=int, default=2,
                        help='Number of LSTM layers')
    parser.add_argument('--use-attention', action='store_true',
                        help='Use self-attention mechanism')
    parser.add_argument('--num-attention-heads', type=int, default=4,
                        help='Number of attention heads')
    
    args = parser.parse_args()
    
    # Initialize config with command line arguments
    config = TrainingConfig()
    config.hidden_size = args.hidden_size
    config.num_layers = args.num_layers
    config.use_attention = args.use_attention
    config.num_attention_heads = args.num_attention_heads
    config.patience = 15  # Early stopping patience
    
    # Set device
    if args.device is not None:
        config.device = torch.device(args.device)
    else:
        config.device = torch.device('cuda' if torch.cuda.is_available() 
                                  else 'mps' if torch.backends.mps.is_available() 
                                  else 'cpu')
    
    print(f"Using device: {config.device}")
    
    # Train the model
    metrics = train(
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        num_workers=args.num_workers,
        device=args.device,
        config=config
    )
    
    print("\nTraining completed!")
    print(f"Best validation accuracy: {metrics['best_val_accuracy']:.2f}%")
    print(f"Test accuracy: {metrics['test_accuracy']:.2f}%")
    print(f"Model and metrics saved to: {os.path.abspath(args.model_dir)}")
    
    # Verify output structure using the config object
    verify_output_structure(config)
    
    return metrics


if __name__ == '__main__':
    # Set the multiprocessing start method to 'spawn' for macOS/Windows compatibility.
    # This guard is essential to prevent issues with child processes.
    try:
        torch.multiprocessing.set_start_method('spawn')
    except RuntimeError:
        # This might be raised if the method has already been set.
        pass

    # Run the main training function
    main()
