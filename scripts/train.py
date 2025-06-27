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
from src.segmentation.utils.dataset_utils import split_dataset, load_config, show_augmentation_samples

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
