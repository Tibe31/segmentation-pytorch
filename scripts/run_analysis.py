from PIL import Image
import numpy as np
from run_inference import run_inference
from utils import load_torch_model
import os
import glob
import torch
from torchvision import transforms


def simple_inference(image_path, model, outputs_path):
    """
    Simple inference function that processes a single image using existing run_inference
    """
    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    image = np.array(image)
    
    # Use the existing run_inference function
    # We pass the image as a single chunk in a list
    chunks = [image]
    outputs = run_inference(chunks, batch_size=1, model=model)
    
    # Get the output (should be a list with one element)
    output_binary = outputs[0]
    
    # Save the result
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_filename = os.path.join(outputs_path, f"{base_name}_prediction.png")
    
    # Convert to PIL and save (output_binary is already 0/1, multiply by 255 for visualization)
    output_image = Image.fromarray((output_binary * 255).astype(np.uint8))
    output_image.save(output_filename)
    
    print(f"Processed {image_path} -> {output_filename}")


if __name__ == "__main__":
    (
        model,
        images_path,
        outputs_path,
    ) = load_torch_model()
    
    # Create outputs directory if it doesn't exist
    os.makedirs(outputs_path, exist_ok=True)
    
    # Get all image files from the images_path directory
    image_files = glob.glob(os.path.join(images_path, "*.png"))
    
    print(f"Found {len(image_files)} images to process")
    
    for image_path in image_files:
        try:
            simple_inference(
                image_path,
                model,
                outputs_path,
            )
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
