# Segmentation PyTorch

## Project Overview
This repository provides a pipeline for semantic segmentation using PyTorch Lightning and segmentation models. It includes scripts for dataset preparation, training, ONNX export, and inference.

## Dataset Structure
The dataset should be organized as follows after running the split script:

```
dataset_root/
  train/
    images/
      img1.png
      img2.png
      ...
    masks/
      img1.png
      img2.png
      ...
  val/
    images/
    masks/
  test/
    images/
    masks/
```
- Each split (`train`, `val`, `test`) contains an `images/` folder with input images and a `masks/` folder with the corresponding masks. The mask filenames must match the image filenames.

## Dataset Splitting
Dataset splitting is performed **automatically** when you launch training with `scripts/train.py`. The utility functions for splitting (`split_dataset`) and config loading (`load_config`) are located in `src/segmentation/utils/dataset_utils.py`.

A CLI script (`scripts/split_dataset.py`) is also available if you want to manually split the dataset (for example, to inspect or modify the splits before training):

```bash
python scripts/split_dataset.py
```
This will create the `train/`, `val/`, and `test/` folders with the correct structure inside the directory specified by `data.splits_dir` in your config.

## Training
Update your `configs/config.yaml` to point to the new split folders:
- `data.splits_dir`: path to the folder containing `train/`, `val/`, `test/` (e.g., `dataset_creation/`)
- `data.images_dir` and `data.masks_dir` are only needed for the initial split, not for training.

To start training:
```bash
python scripts/train.py
```
The script will check if the split folders exist and run the split if needed.

## ONNX Export
To export your trained model to ONNX format:
```bash
python onnx/convert_to_onnx.py
```

## Inference

### PyTorch Inference

To run inference with a trained PyTorch model (`.ckpt`), use the `scripts/predict.py` script.

First, ensure your `configs/config.yaml` is configured correctly. The script will load the model and paths from the `inference` section of this file:

```yaml
inference:
  model_path: "checkpoints/best_model.ckpt" # Path to your .ckpt model
  input_images_dir: "path/to/your/images"   # Directory with images for batch mode
  outputs_dir: "path/to/save/predictions"  # Directory to save the output masks
```

**Batch Mode (Default):**
To predict masks for all images in the `input_images_dir` specified in your config. This is the default mode, so you can run it without specifying `--mode`.
```bash
python scripts/predict.py
```
Or explicitly:
```bash
python scripts/predict.py --mode batch
```

**Single Image Mode:**
To predict a mask for a single image, specify its path using the `--image` argument:
```bash
python scripts/predict.py --mode single --image "path/to/your/single_image.png"
```

### ONNX Inference
See `onnx/onnx_predict.py` for ONNX inference.

## Notes
- The code expects mask filenames to match image filenames.
- The DataModule now loads data directly from the split folders, not from .txt files.
- The dataset split logic is reusable and located in `src/segmentation/utils/dataset_utils.py`.
- The CLI split script is optional and mainly for manual inspection or custom splitting.

---
For any questions or issues, please open an issue on the repository.
