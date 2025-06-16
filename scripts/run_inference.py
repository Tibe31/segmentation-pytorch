import torch
from torchvision import transforms
import torch.nn.functional as F
import halcon as ha
import numpy as np
import math
import cv2 as cv
import time
from line_profiler_pycharm import profile


@profile
def run_inference(chunks, batch_size, model):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            # Normalize if used during training
            transforms.Resize((512, 512)),
        ]
    )
    start = time.time()
    outputs = []
    # Split the chunks into smaller batches of the specified batch size
    for i in range(0, len(chunks), batch_size):

        # Get the current batch of chunks
        batch = chunks[i : i + batch_size]
        # Apply preprocessing transformation if needed
        if transform:
            # For batch, normalization should be applied individually per image

            batch_tensor = torch.stack([transform(image) for image in batch])

            # Perform inference on the whole batch
        with torch.no_grad():  # No need to compute gradients

            output = model(batch_tensor.cuda())
        start_ = time.time()
        output = output.sigmoid()
        output = F.interpolate(
            output, size=(600, 600), mode="bilinear", align_corners=False
        )

        output_binary = output[:, 0].cpu().numpy()

        output_binary[output_binary > 0.5] = 1
        output_binary[output_binary <= 0.5] = 0
        outputs.append(output_binary)
        end_ = time.time()
        print("post_process time {}".format(end_ - start_))
    end = time.time()
    print("Inference time {}".format(end - start))
    return outputs


def extract_crop_and_run_inference(image_polar, parameters, model, is_back):
    border_external = ha.himage_as_numpy_array(image_polar)
    crop_width = 600
    crop_height = 200
    num_chunks = (
        border_external.shape[1] // crop_width
    )  # This should be 12 in your case
    chunks = []

    # Split the image into chunks of width 600 and height 200
    for i in range(0, num_chunks, 3):  # Skip in steps of 3
        # Stack three consecutive chunks vertically
        chunk = np.vstack(
            [
                border_external[:, i * crop_width : (i + 1) * crop_width]
                for i in range(i, i + 3)
            ]
        )
        chunks.append(chunk)
    single_mask = []
    outputs_external = run_inference(chunks, batch_size=2, model=model)

    for single_batch in outputs_external:
        for single_mask_output in single_batch:
            for i in range(3):
                single_mask.append(
                    single_mask_output[i * crop_height : crop_height * (i + 1), :]
                )
    rectified_output = np.hstack(single_mask)
    if is_back:
        radius = parameters["r_filter"] + 200
    else:
        radius = parameters["r_filter"]
    circumference_filter = int(2 * math.pi * radius)
    rectified_output = rectified_output[:, :circumference_filter]
    num_labels, labels = cv.connectedComponents(rectified_output.astype(np.uint8))
    filtered_mask = np.zeros_like(rectified_output)

    for label in range(1, num_labels):
        component_mask = (labels == label).astype(np.uint8)
        x, y, w, h = cv.boundingRect(component_mask)
        if is_back:
            if y + h == crop_height:
                # If the min_y is crop_height (200), keep the component in the filtered mask
                filtered_mask[component_mask == 1] = 1
        else:
            if y == 0 and h > 10 and np.sum(component_mask) / label > 1000:
                filtered_mask[component_mask == 1] = 1
    return filtered_mask
