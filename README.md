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
To split your dataset into train/val/test folders:
1. Place all your images in a single folder (e.g., `all_images/`) and all masks in another folder (e.g., `all_masks/`).
2. Set the paths in `configs/config.yaml` under `data.images_dir` and `data.masks_dir` to point to these folders.
3. Set `data.splits_dir` in the config to the desired output root for the split folders (e.g., `dataset_creation/`).
4. Run:
   ```bash
   python scripts/split_dataset.py
   ```
   This will create the `train/`, `val/`, and `test/` folders with the correct structure inside `splits_dir`.

## Training
Update your `configs/config.yaml` to point to the new split folders:
- `data.splits_dir`: path to the folder containing `train/`, `val/`, `test/` (e.g., `dataset_creation/`)
- `data.images_dir` and `data.masks_dir` are only needed for the initial split, not for training.

To start training:
```bash
python scripts/train.py
```

## ONNX Export
To export your trained model to ONNX format:
```bash
python onnx/convert_to_onnx.py
```

## Inference
See `onnx/onnx_predict.py` for ONNX inference or use the provided scripts for PyTorch inference.

## Notes
- The code expects mask filenames to match image filenames.
- The DataModule now loads data directly from the split folders, not from .txt files.

---
For any questions or issues, please open an issue on the repository.
