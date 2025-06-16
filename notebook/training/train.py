import os
import pytorch_lightning as pl
from dotenv import load_dotenv, find_dotenv
from pytorch_lightning.callbacks import ModelCheckpoint
from src.model.segmentation import SegmentationModels
from src.datamodule.segmentation import SegmentationDataModule

from pytorch_lightning.loggers import CometLogger

load_dotenv(find_dotenv())

if __name__ == "__main__":

    #comet_logger = CometLogger(
    #    api_key=os.environ.get("API_KEY"),
    #    project_name=os.environ.get("PROJECT_NAME"),
    #    workspace=os.environ.get("WORKSPACE"),
    #)
    model = SegmentationModels("fpn", "resnet34", in_channels=3, out_classes=1)
    data_module = SegmentationDataModule(
        txt_split="dataset_creation", batch_size=4, resize_size=(512, 512)
    )
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
