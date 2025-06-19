import os
import sys
# Ensure project root is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import random
import yaml
from pathlib import Path

"""
Script to split a dataset of images into train, val, and test splits.
Writes train.txt, val.txt, and test.txt in the specified split directory.
Each file contains only filenames (one per line).
"""

def load_config(config_path):
    """
    Load a YAML configuration file.
    Args:
        config_path (str): Path to the YAML config file.
    Returns:
        dict: Parsed configuration dictionary.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def split_dataset(config, ext='.png', seed=42):
    """
    Split the dataset into train, val, and test splits and write split files.
    Args:
        config (dict): Configuration dictionary containing:
            - data.images_dir: Directory containing images
            - data.splits_dir: Directory to save split .txt files
        ext (str): File extension to filter images (default: '.png')
        seed (int): Random seed for reproducibility
    """
    # Extract paths from config
    images_dir = config['data']['images_dir']
    split_dir = config['data']['splits_dir']
    
    # Get split ratios from config or use defaults
    split_config = config.get('split_ratios', {
        'train': 0.7,
        'val': 0.15,
        'test': 0.15
    })
    train_ratio = split_config.get('train', 0.7)
    val_ratio = split_config.get('val', 0.15)
    test_ratio = split_config.get('test', 0.15)

    # Create split directory
    os.makedirs(split_dir, exist_ok=True)
    
    # Get and shuffle files
    all_files = [f for f in os.listdir(images_dir) if f.endswith(ext)]
    random.seed(seed)
    random.shuffle(all_files)
    
    # Calculate split sizes
    n_total = len(all_files)
    n_train = int(train_ratio * n_total)
    n_val = int(val_ratio * n_total)
    n_test = n_total - n_train - n_val
    
    # Split files
    train_files = all_files[:n_train]
    val_files = all_files[n_train:n_train+n_val]
    test_files = all_files[n_train+n_val:]
    
    # Write split files with full image paths
    with open(os.path.join(split_dir, "train.txt"), "w") as f:
        f.write("\n".join([os.path.join(images_dir, fname) for fname in train_files]))
    with open(os.path.join(split_dir, "val.txt"), "w") as f:
        f.write("\n".join([os.path.join(images_dir, fname) for fname in val_files]))
    with open(os.path.join(split_dir, "test.txt"), "w") as f:
        f.write("\n".join([os.path.join(images_dir, fname) for fname in test_files]))
    
    print(f"Splits created in {split_dir}!")
    print(f"Train: {len(train_files)} images ({train_ratio*100:.1f}%)")
    print(f"Val: {len(val_files)} images ({val_ratio*100:.1f}%)")
    print(f"Test: {len(test_files)} images ({test_ratio*100:.1f}%)")

if __name__ == "__main__":
    # Calcola il path assoluto del config
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'configs', 'config.yaml'))
    config = load_config(config_path)
    split_dataset(config) 