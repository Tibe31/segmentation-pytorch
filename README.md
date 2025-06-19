# BINARY SEGMENTATION USING PYTORCH

This repository contains code for training, inference, and analysis of binary segmentation models (e.g., for detecting growth on filter plates) using PyTorch Lightning. It is also possible to export trained model to onnx and execute predictions on it. 

## Project Structure

```
segmentation-pytorch/
├── src/segmentation/
│   ├── models/         # Models (Unet, FPN, ...)
│   ├── data/           # DataModule, Dataset, Transforms
│   └── utils/          # Utilities (metrics, visualization, ...)
├── scripts/            # CLI scripts for training, inference, dataset splitting
├── configs/            # Centralized YAML configuration
├── checkpoints/        # Saved models
├── outputs/            # Prediction outputs
├── onnx/               # ONNX model conversion and inference
└── README.md
```

## Installing Dependencies

Dependencies are managed with poetry:
```bash
pip install poetry
poetry install
```
Tested with Python 3.11 and CUDA 12.6. Update the configuration if needed for your system.

## Configuration

All configuration (model, paths, augmentation, resize, etc.) is centralized in `configs/config.yaml`.

**Note on relative and absolute paths:**
- You can use either absolute or relative paths in `configs/config.yaml` for all entries (`images_dir`, `masks_dir`, `splits_dir`, etc.).
- If you use a relative path, it will always be interpreted relative to the project root, regardless of the folder from which you launch the scripts.
- Example of a relative path:
  ```yaml
  data:
    images_dir: data/images
    masks_dir: data/masks
    splits_dir: dataset_creation
  ```
- Example of an absolute path:
  ```yaml
  data:
    images_dir: C:/Users/Andrea/Desktop/images
    masks_dir: C:/Users/Andrea/Desktop/masks
    splits_dir: C:/Users/Andrea/Desktop/segmentation-pytorch/dataset_creation
  ```

Edit this file to:
- Change architecture, backbone, parameters
- Update image, mask, output, and checkpoint paths
- Customize augmentations (see example and comments in the file)
- Set the resize size for both training and inference:
  ```yaml
  model:
    input_size: [512, 512]  # (height, width)
  ```
- Set data directories and split folder:
  ```yaml
  data:
    images_dir: /path/to/images
    masks_dir: /path/to/masks
    splits_dir: /path/to/split_folder  # contains train.txt, val.txt, test.txt
  ```
- Set the split ratios (optional):
  ```yaml
  split_ratios:
    train: 0.7
    val: 0.15
    test: 0.15
  ```
- Each split file (e.g. `train.txt`) must contain only filenames, one per line:
  ```
  img_001.png
  img_002.png
  ...
  ```
  The full path will be constructed as `images_dir/filename` and `masks_dir/filename`.

## Dataset Splitting

To create the split files (train/val/test) based on the configuration:
```bash
python scripts/split_dataset.py
```
The script will automatically read paths and split ratios from the YAML config.

## Training

To start training:
```bash
python scripts/train.py
```

## Inference (Prediction)

All inference modes are unified in `scripts/predict.py`.

### Batch prediction (entire folder):
```bash
python scripts/predict.py --mode batch
```
Processes all PNG images in the folder specified by `inference.input_images_dir` in your config.

### Single image prediction:
```bash
python scripts/predict.py --mode single --image path/to/image.png
```
Processes only the specified image and saves the result in the output folder from the config.

- The resize size used for inference is always read from `model.input_size` in the config.
- You can specify a custom config file with `--config path/to/your_config.yaml`.

## Notes
- All parameters and paths are centralized in `configs/config.yaml`.
- You can add new models, datasets, or augmentation pipelines by editing only the structure in `src/segmentation/` and the YAML config.
- Notebooks are for exploration only and do not contain business logic.

---
For questions or suggestions, open an issue or contact the maintainers.
