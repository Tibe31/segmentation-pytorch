import os
import sys
# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytorch_lightning as pl
from dotenv import load_dotenv, find_dotenv
from pytorch_lightning.callbacks import ModelCheckpoint
from src.model.segmentation import SegmentationModels
from src.datamodule.segmentation import SegmentationDataModule
from src.dataset.segmentation import SegmentationDataset
import matplotlib.pyplot as plt
import numpy as np

from pytorch_lightning.loggers import CometLogger

load_dotenv(find_dotenv())

def show_augmentation_samples(data_module):
    """Show augmented images and their corresponding masks to verify augmentation settings"""
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

if __name__ == "__main__":

    #comet_logger = CometLogger(
    #    api_key=os.environ.get("API_KEY"),
    #    project_name=os.environ.get("PROJECT_NAME"),
    #    workspace=os.environ.get("WORKSPACE"),
    #)
    model = SegmentationModels("fpn", "resnet34", in_channels=3, out_classes=1)
    data_module = SegmentationDataModule(
        txt_split="dataset_creation", batch_size=1, resize_size=(512, 512)
    )
    
    # Show augmentation samples before training
    print("Showing augmentation samples...")
    show_augmentation_samples(data_module)
    
    checkpoint_callback = ModelCheckpoint(
        monitor="valid_loss",  # The metric to monitor
        dirpath="checkpoints/",  # Directory to save the model
        filename="best_model",  # The name format for the checkpoint file
        save_top_k=1,  # Save only the best model (based on the validation loss)
        mode="min",  # 'min' because we want to minimize validation loss
        verbose=True,  # Print out information when saving the model
    )
    trainer = pl.Trainer(
        max_epochs=550, callbacks=[checkpoint_callback])#, logger=comet_logger

    # Train the model
    trainer.fit(model, datamodule=data_module)
