import os
import glob
from PIL import Image
import numpy as np
import onnxruntime
import torch
from torchvision import transforms
import yaml
import argparse

from src.segmentation.utils.postprocessing import postprocess_binary_mask

def preprocess_image(image_path, resize_size):
    """
    Preprocess an image for ONNX inference.
    Steps:
      - Open and convert to RGB
      - ToTensor
      - Normalize with ImageNet mean/std
      - Resize to the given size
    Args:
        image_path (str): Path to the input image
        resize_size (tuple): (height, width) for resizing
    Returns:
        np.ndarray: Preprocessed image as numpy array, shape [1, C, H, W]
    """
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        transforms.Resize(resize_size)
    ])
    tensor = transform(image).unsqueeze(0)  # [1, C, H, W]
    return tensor.numpy()

def postprocess_output(
    output,
    output_size,
    threshold=0.5,
    resize_strategy="resize_then_threshold",
):
    """
    Postprocess ONNX model output to obtain a binary mask image.
    Steps:
      - Resize probabilities then threshold, or threshold then resize
        depending on the configured strategy
    Args:
        output (np.ndarray): Raw ONNX model output
        output_size (tuple): (height, width) for the final mask
        threshold (float): Probability threshold for the positive class
        resize_strategy (str): Postprocessing strategy
    Returns:
        PIL.Image: Binary mask image
    """
    pred_tensor = torch.from_numpy(output[0]).float()
    binary_mask = postprocess_binary_mask(
        predictions=pred_tensor,
        output_size=output_size,
        threshold=threshold,
        resize_strategy=resize_strategy,
        from_logits=True,
    )
    mask = (binary_mask[0, 0].numpy() * 255).astype(np.uint8)
    mask_img = Image.fromarray(mask)
    return mask_img

def run_onnx_inference(
    image_path,
    ort_session,
    output_dir,
    resize_size,
    threshold=0.5,
    resize_strategy="resize_then_threshold",
):
    """
    Run ONNX inference on a single image and save the predicted mask.
    Args:
        image_path (str): Path to the input image
        ort_session (onnxruntime.InferenceSession): ONNX session
        output_dir (str): Directory to save the output mask
        resize_size (tuple): (height, width) for model input resizing
        threshold (float): Probability threshold for the positive class
        resize_strategy (str): Postprocessing strategy
    """
    original_size = Image.open(image_path).size[::-1]
    input_tensor = preprocess_image(image_path, resize_size)
    ort_inputs = {"input": input_tensor}
    ort_outputs = ort_session.run(None, ort_inputs)
    mask = postprocess_output(
        ort_outputs,
        original_size,
        threshold=threshold,
        resize_strategy=resize_strategy,
    )
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(output_dir, f"{base_name}_onnx_prediction.png")
    mask.save(output_path)
    print(f"✅ {image_path} -> {output_path}")

if __name__ == "__main__":
    """
    ONNX batch inference entrypoint.
    Loads config, model, and runs inference on all PNG images in the input folder.
    """
    parser = argparse.ArgumentParser(description="ONNX inference script with config support")
    parser.add_argument('--config', type=str, default='../configs/config.yaml', help='Path to config YAML')
    parser.add_argument('--onnx_model', type=str, default='unet_resnet34.onnx', help='Path to ONNX model')
    parser.add_argument('--images_dir', type=str, help='Override images dir')
    parser.add_argument('--outputs_dir', type=str, default='./onnx_outputs', help='Output dir')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    resize_size = tuple(config['model']['input_size'])
    inference_cfg = config.get('inference', {})
    images_dir = args.images_dir or config['inference']['input_images_dir']
    outputs_dir = args.outputs_dir
    onnx_model_path = args.onnx_model
    threshold = inference_cfg.get('threshold', 0.5)
    resize_strategy = inference_cfg.get('resize_strategy', 'resize_then_threshold')

    os.makedirs(outputs_dir, exist_ok=True)
    ort_session = onnxruntime.InferenceSession(
        onnx_model_path,
        providers=["CPUExecutionProvider"]
    )
    image_files = glob.glob(os.path.join(images_dir, "*.png"))
    print(f"Found {len(image_files)} images in '{images_dir}'")
    for img_path in image_files:
        run_onnx_inference(
            img_path,
            ort_session,
            outputs_dir,
            resize_size,
            threshold=threshold,
            resize_strategy=resize_strategy,
        )
