# Segmentation PyTorch

Binary semantic segmentation pipeline built on PyTorch Lightning and [segmentation_models_pytorch](https://github.com/qubvel/segmentation_models.pytorch). Covers dataset preparation, training, ONNX export, and inference.

The model architecture is configurable: any encoder-decoder combination supported by `segmentation_models_pytorch` (UNet, FPN, DeepLabV3+, etc.) can be selected from `configs/config.yaml` without touching the code.

## Project Structure

```
segmentation-pytorch/
  configs/
    config.yaml                  # all settings (model, training, data paths, augmentation)
  scripts/
    train.py                     # training entrypoint
    predict.py                   # PyTorch inference (batch or single image)
    split_dataset.py             # manual dataset split (optional)
  onnx/
    convert_to_onnx.py           # export trained model to ONNX
    onnx_predict.py              # ONNX inference
  src/segmentation/
    models/unet.py               # LightningModule (model, loss, metrics, optimizer)
    data/dataset.py              # PyTorch Dataset
    data/datamodule.py           # LightningDataModule (loaders, augmentation)
    utils/dataset_utils.py       # split logic, config loading, augmentation preview
    utils/utils.py               # model loading helper for inference
```

## Installation

Requires Python 3.9+.

```bash
git clone <repo-url>
cd segmentation-pytorch
pip install -r requirements.txt
```

The `requirements.txt` ships with PyTorch built for **CUDA 11.8**. If you need a different CUDA version or CPU-only, install PyTorch manually first following the [official instructions](https://pytorch.org/get-started/locally/) and then install the remaining dependencies.

## Configuration

Everything is driven by `configs/config.yaml`. The shipped config contains **absolute paths that you must update** to match your machine.

### Model

```yaml
model:
  input_size: [512, 512]    # resize dimensions (height, width)
  architecture: fpn          # any smp architecture: unet, fpn, deeplabv3plus, ...
  backbone: resnet34         # any timm / smp encoder: resnet34, efficientnet-b0, ...
  in_channels: 3             # input channels (3 for RGB)
  out_classes: 1             # output classes (1 for binary segmentation)
```

### Training

```yaml
training:
  epochs: 550
  batch_size: 1
  lr: 0.001
  num_workers: 2
```

### Data Paths

```yaml
data:
  splits_dir: path/to/dataset_creation   # folder with train/, val/, test/
  images_dir: path/to/raw_dataset/images  # only for initial split
  masks_dir: path/to/raw_dataset/masks    # only for initial split

split_ratios:
  train: 0.7
  val: 0.15
  test: 0.15
```

`images_dir` and `masks_dir` are only used by the split step. Once the splits exist, only `splits_dir` matters.

### Checkpointing

```yaml
output:
  checkpoint_dir: checkpoints/
  checkpoint_name: best_model
  monitor: valid_loss
  save_top_k: 1
  mode: min
```

### Augmentation

The augmentation pipeline is defined declaratively in the config using [Albumentations](https://albumentations.ai/) transform names. Transforms are applied in order; `OneOf` and nested `Compose` blocks are supported.

```yaml
augmentation:
  - name: HorizontalFlip
    p: 0.9
  - name: OneOf
    transforms:
      - name: RandomBrightnessContrast
        p: 0.6
      - name: RandomGamma
        p: 0.4
    p: 0.8
  - name: Blur
    p: 0.5
  - name: GridDropout
    p: 0.2
    ratio: 0.3
    unit_size_range: [10, 20]
    fill: random_uniform
```

Any Albumentations transform can be added by name with its parameters. See the comments inside `config.yaml` for more examples.

Set `show_augmentation_samples: true` in the config to display a visual preview of augmented image/mask pairs before training starts.

## Dataset Structure

After splitting, the data directory should look like this:

```
splits_dir/
  train/
    images/
    masks/
  val/
    images/
    masks/
  test/
    images/
    masks/
```

Mask filenames must match the corresponding image filenames. All images are expected to be `.png`.

## Dataset Splitting

Splitting is performed **automatically** the first time you run `scripts/train.py` (if the split folders don't exist yet).

You can also split manually:

```bash
python scripts/split_dataset.py
```

The ratios and paths are read from `config.yaml`.

## Training

```bash
python scripts/train.py
```

The script loads the config, splits the dataset if needed, optionally shows augmentation samples, and starts training. The best checkpoint is saved according to the `output` section of the config.

## ONNX Export

Exports the trained model to ONNX with a fixed spatial resolution (from `model.input_size`) and dynamic batch size:

```bash
python onnx/convert_to_onnx.py
```

The checkpoint path is read from `inference.model_path` in the config. The output `.onnx` file is saved under `onnx/`.

## Inference

### PyTorch

```bash
# batch: process all images in the configured input_images_dir
python scripts/predict.py

# single image
python scripts/predict.py --mode single --image "path/to/image.png"
```

Both modes read `inference.model_path`, `inference.input_images_dir`, and `inference.outputs_dir` from the config. You can override the config path with `--config path/to/other_config.yaml`.

### ONNX

```bash
python onnx/onnx_predict.py --onnx_model onnx/model.onnx
```

Optional arguments:

| Flag           | Default                  | Description                     |
|----------------|--------------------------|---------------------------------|
| `--config`     | `../configs/config.yaml` | Path to config file             |
| `--onnx_model` | `unet_resnet34.onnx`     | Path to the `.onnx` model       |
| `--images_dir` | from config              | Override input images directory |
| `--outputs_dir`| `./onnx_outputs`         | Output directory                |

## Notes

- Mask filenames must match image filenames exactly.
- The DataModule loads data directly from the split folders, not from `.txt` file lists.
- The split logic in `src/segmentation/utils/dataset_utils.py` is reusable outside the training script.

---
For questions or issues, open an issue on the repository.
