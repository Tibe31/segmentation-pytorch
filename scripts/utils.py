import json
import sys
import os
# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.segmentation import SegmentationModels

def load_torch_model():
    with open("models.json", "r") as model_info:
        models_path = json.load(model_info)

    model = SegmentationModels.load_from_checkpoint(
        models_path["model_path"],
        arch="fpn",
        encoder_name="resnet34",
        in_channels=3,
        out_classes=1,
    )
    model.eval()

    images_path = models_path["images_path"]
    outputs_path = models_path["outputs_path"]
    return (
        model,
        images_path,
        outputs_path,
    )