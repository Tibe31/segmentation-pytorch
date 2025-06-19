import os
import sys
# Ensure project root is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.segmentation.utils.dataset_utils import split_dataset, load_config

"""
CLI script to split a dataset of images and masks into train, val, and test folders.
Each split will have 'images/' and 'masks/' subfolders containing the corresponding files.
Usually, you do not need to run this manually, as it is called automatically by train.py.
"""

if __name__ == "__main__":
    # Entry point for manual split
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'configs', 'config.yaml'))
    config = load_config(config_path)
    split_dataset(config) 