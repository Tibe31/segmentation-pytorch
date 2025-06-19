import os
import sys
# Ensure project root is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import yaml
import pytorch_lightning as pl
from dotenv import load_dotenv, find_dotenv
from pytorch_lightning.callbacks import ModelCheckpoint
from src.segmentation.models.unet import SegmentationModels
from src.segmentation.data.datamodule import SegmentationDataModule
import matplotlib.pyplot as plt
import numpy as np
from pytorch_lightning.loggers import CometLogger
from pathlib import Path
# Import split_dataset and load_config from split_dataset.py
from scripts.split_dataset import split_dataset, load_config

load_dotenv(find_dotenv())

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

def main():
    """
    Main training entrypoint.
    Loads configuration, checks and performs dataset split if needed, initializes model and data module, shows augmentation samples,
    sets up checkpointing and trainer, and starts training.
    """
    # Compute absolute path to config
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'configs', 'config.yaml'))
    config = load_config(config_path)

    # --- Check if dataset is already split, otherwise perform split ---
    splits_dir = config['data']['splits_dir']
    # Check for train/val/test folders with images/ and masks/
    required_folders = [
        os.path.join(splits_dir, split, sub)
        for split in ['train', 'val', 'test']
        for sub in ['images', 'masks']
    ]
    if not all(os.path.isdir(folder) for folder in required_folders):
        print("Split folders not found. Running dataset split...")
        split_dataset(config)
    else:
        print("Split folders found. Skipping dataset split.")

    # Model configuration
    model_cfg = config['model']
    model = SegmentationModels(
        model_cfg.get('architecture', 'fpn'),
        model_cfg.get('backbone', 'resnet34'),
        in_channels=model_cfg.get('in_channels', 3),
        out_classes=model_cfg.get('out_classes', 1)
    )

    # Initialize DataModule with entire config
    data_module = SegmentationDataModule(config)

    # Show augmentation samples before training
    if config.get('show_augmentation_samples', True):
        print("Showing augmentation samples...")
        show_augmentation_samples(data_module)

    # Checkpoint callback configuration
    output_cfg = config['output']
    checkpoint_callback = ModelCheckpoint(
        monitor=output_cfg.get('monitor', 'valid_loss'),
        dirpath=output_cfg.get('checkpoint_dir', 'checkpoints/'),
        filename=output_cfg.get('checkpoint_name', 'best_model'),
        save_top_k=output_cfg.get('save_top_k', 1),
        mode=output_cfg.get('mode', 'min'),
        verbose=True,
    )

    # Trainer configuration
    trainer_cfg = config['training']
    trainer = pl.Trainer(
        max_epochs=trainer_cfg.get('epochs', 550),
        callbacks=[checkpoint_callback],
        # logger=comet_logger if needed
    )

    # Train the model
    trainer.fit(model, datamodule=data_module)

if __name__ == "__main__":
    """
    Entrypoint for training. Loads config, checks dataset split, and starts the training pipeline.
    """
    main()
