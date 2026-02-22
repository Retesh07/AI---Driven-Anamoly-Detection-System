#!/usr/bin/env python3
"""
Script to prepare RWF-2000 dataset splits.

This script:
1. Scans the RWF-2000 dataset directory
2. Labels each video based on its parent folder (Fight/NonFight)
3. Creates stratified train/val/test splits
4. Saves the splits to a master CSV file

Usage:
    python 1_prepare_splits.py --input_dir /path/to/RWF-2000 --output_file master_splits.csv
"""

import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import defaultdict
import random

# Set random seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

def scan_dataset(dataset_dir):
    """
    Scan the RWF-2000 dataset directory and return a list of video paths with labels.
    
    Args:
        dataset_dir (str): Path to the RWF-2000 dataset root directory
        
    Returns:
        list: List of tuples containing (video_path, label)
    """
    video_paths = []
    
    # Expected subdirectories
    for split in ['train', 'val']:
        for label in ['Fight', 'NonFight']:
            video_dir = os.path.join(dataset_dir, split, label)
            if not os.path.exists(video_dir):
                print(f"Warning: Directory not found: {video_dir}")
                continue
                
            # Get all video files
            for video_file in os.listdir(video_dir):
                if video_file.lower().endswith(('.mp4', '.avi', '.mov')):
                    video_path = os.path.join(video_dir, video_file)
                    video_paths.append({
                        'path': video_path,
                        'label': 1 if label == 'Fight' else 0,  # 1 for Fight, 0 for NonFight
                        'split': split,
                        'filename': video_file
                    })
    
    return video_paths

def create_splits(video_data, test_size=0.2, val_size=0.1, random_state=42):
    """
    Create stratified train/val/test splits.
    
    Args:
        video_data (list): List of video data dictionaries
        test_size (float): Proportion of data for test set
        val_size (float): Proportion of training data for validation set
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (train_data, val_data, test_data)
    """
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(video_data)
    
    # If the dataset already has predefined splits, use them
    if 'split' in df.columns and all(split in df['split'].unique() for split in ['train', 'val']):
        print("Using predefined splits from dataset...")
        train_val_df = df[df['split'] == 'train'].copy()
        test_df = df[df['split'] == 'val'].copy()
        
        # Split train into train/val
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_size,
            stratify=train_val_df['label'],
            random_state=random_state
        )
    else:
        # If no predefined splits, create them from scratch
        print("Creating new splits...")
        # First split: separate test set
        train_val_df, test_df = train_test_split(
            df,
            test_size=test_size,
            stratify=df['label'],
            random_state=random_state
        )
        
        # Second split: separate validation set from training
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_size/(1-test_size),  # Adjust val_size to be relative to train+val
            stratify=train_val_df['label'],
            random_state=random_state
        )
    
    return train_df.to_dict('records'), val_df.to_dict('records'), test_df.to_dict('records')

def save_splits_to_csv(splits, output_file):
    """
    Save the dataset splits to a CSV file.
    
    Args:
        splits (dict): Dictionary containing train/val/test splits
        output_file (str): Path to the output CSV file
    """
    # Combine all splits into a single DataFrame
    all_data = []
    for split_name, split_data in splits.items():
        for item in split_data:
            all_data.append({
                'filename': item['filename'],
                'path': item['path'],
                'label': item['label'],
                'split': split_name
            })
    
    # Save to CSV
    df = pd.DataFrame(all_data)
    df.to_csv(output_file, index=False)
    print(f"Saved {len(df)} samples to {output_file}")
    
    # Print dataset statistics
    print("\nDataset Statistics:")
    print("=" * 50)
    print(f"Total samples: {len(df)}")
    print(f"Fight samples: {len(df[df['label'] == 1])} ({len(df[df['label'] == 1])/len(df)*100:.1f}%)")
    print(f"Non-Fight samples: {len(df[df['label'] == 0])} ({len(df[df['label'] == 0])/len(df)*100:.1f}%)")
    
    print("\nSplit Distribution:")
    print("-" * 50)
    for split in ['train', 'val', 'test']:
        split_df = df[df['split'] == split]
        print(f"{split.upper()}:")
        print(f"  Total: {len(split_df)} samples")
        print(f"  Fight: {len(split_df[split_df['label'] == 1])} samples")
        print(f"  Non-Fight: {len(split_df[split_df['label'] == 0])} samples")
        print()

def main():
    parser = argparse.ArgumentParser(description='Prepare RWF-2000 dataset splits')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Path to the RWF-2000 dataset root directory')
    parser.add_argument('--output_file', type=str, default='master_splits.csv',
                        help='Output CSV file path (default: master_splits.csv)')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Proportion of data for test set (default: 0.2)')
    parser.add_argument('--val_size', type=float, default=0.1,
                        help='Proportion of training data for validation (default: 0.1)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    print(f"Scanning dataset directory: {args.input_dir}")
    video_data = scan_dataset(args.input_dir)
    
    if not video_data:
        print("Error: No video files found in the specified directory.")
        return
    
    print(f"Found {len(video_data)} video files.")
    
    # Create splits
    train_data, val_data, test_data = create_splits(
        video_data,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.seed
    )
    
    # Save splits to CSV
    splits = {
        'train': train_data,
        'val': val_data,
        'test': test_data
    }
    
    save_splits_to_csv(splits, args.output_file)
    print("\nDataset preparation complete!")

if __name__ == '__main__':
    main()
