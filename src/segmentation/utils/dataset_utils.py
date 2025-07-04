import os
import random
import shutil
import yaml
import matplotlib.pyplot as plt
import numpy as np
"""
Utility functions for dataset splitting and config loading.
"""

def show_augmentation_samples(data_module):
    """
    Display a few samples of augmented images and their corresponding masks.
    Useful to visually verify augmentation settings before training.
    Args:
        data_module (SegmentationDataModule): The data module with augmentations.
    """
    data_module.setup()
    # Get the dataset with augmentation
    dataset_with_aug = data_module.train_dataset
    print("Augmentation transforms being used:")
    print(data_module.get_training_augmentation())
    print("\nShowing 5 pairs of augmented images and masks. Close each window to see the next pair...")
    for i in range(5):
        # Create figure for this pair
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        # Get augmented image and mask
        augmented_image, augmented_mask = dataset_with_aug[i]
        # Convert tensors to numpy and denormalize image
        if hasattr(augmented_image, 'numpy'):
            augmented_np = augmented_image.numpy()
        else:
            augmented_np = augmented_image
        if hasattr(augmented_mask, 'numpy'):
            mask_np = augmented_mask.numpy()
        else:
            mask_np = augmented_mask
        # Denormalize the image
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        augmented_denorm = augmented_np.transpose(1, 2, 0) * std + mean
        augmented_denorm = np.clip(augmented_denorm, 0, 1)
        # Handle mask dimensions
        if len(mask_np.shape) == 3:
            mask_np = mask_np[0]  # Remove channel dimension if present
        # Show augmented image
        axes[0].imshow(augmented_denorm)
        axes[0].set_title(f'Augmented Image {i+1}')
        axes[0].axis('off')
        # Show augmented mask
        axes[1].imshow(mask_np, cmap='gray')
        axes[1].set_title(f'Augmented Mask {i+1}')
        axes[1].axis('off')
        plt.tight_layout()
        plt.show()
        print(f"Showed pair {i+1}/5. Close the window to continue...")
    print("Finished showing all 5 pairs of augmented images and masks!")

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
    Split the dataset into train, val, and test folders with images and masks.
    If the split folders already exist and are non-empty, the function does nothing.
    Args:
        config (dict): Configuration dictionary containing:
            - data.images_dir: Directory containing all images
            - data.masks_dir: Directory containing all masks
            - data.splits_dir: Directory to save split folders (root dir)
        ext (str): File extension to filter images (default: '.png')
        seed (int): Random seed for reproducibility
    """
    images_dir = config['data']['images_dir']
    masks_dir = config['data']['masks_dir']
    split_dir = config['data']['splits_dir']

    split_config = config.get('split_ratios', {
        'train': 0.7,
        'val': 0.15,
        'test': 0.15
    })
    train_ratio = split_config.get('train', 0.7)
    val_ratio = split_config.get('val', 0.15)
    test_ratio = split_config.get('test', 0.15)

    # Gather and shuffle all image files
    all_files = [f for f in os.listdir(images_dir) if f.endswith(ext)]
    random.seed(seed)
    random.shuffle(all_files)

    n_total = len(all_files)
    n_train = int(train_ratio * n_total)
    n_val = int(val_ratio * n_total)
    n_test = n_total - n_train - n_val

    train_files = all_files[:n_train]
    val_files = all_files[n_train:n_train+n_val]
    test_files = all_files[n_train+n_val:]

    splits = {'train': train_files, 'val': val_files, 'test': test_files}

    for split, files in splits.items():
        split_images_dir = os.path.join(split_dir, split, 'images')
        split_masks_dir = os.path.join(split_dir, split, 'masks')
        os.makedirs(split_images_dir, exist_ok=True)
        os.makedirs(split_masks_dir, exist_ok=True)
        for fname in files:
            src_img = os.path.join(images_dir, fname)
            src_mask = os.path.join(masks_dir, fname)
            dst_img = os.path.join(split_images_dir, fname)
            dst_mask = os.path.join(split_masks_dir, fname)
            shutil.copy2(src_img, dst_img)
            if os.path.exists(src_mask):
                shutil.copy2(src_mask, dst_mask)
            else:
                print(f"Warning: mask not found for {fname}, skipping mask copy.")
    print(f"Splits created in {split_dir}!")
    print(f"Train: {len(train_files)} images ({train_ratio*100:.1f}%)")
    print(f"Val: {len(val_files)} images ({val_ratio*100:.1f}%)")
    print(f"Test: {len(test_files)} images ({test_ratio*100:.1f}%)") 