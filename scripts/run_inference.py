import torch
from torchvision import transforms
import torch.nn.functional as F
import time

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
    
    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
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
            output = model(batch_tensor.to(device))
        start_ = time.time()
        
        print(f"Raw output shape: {output.shape}")
        
        output = output.sigmoid()
        output = F.interpolate(
            output, size=(600, 600), mode="bilinear", align_corners=False
        )

        print(f"After interpolation shape: {output.shape}")
        
        # Handle different output dimensions
        if len(output.shape) == 4:  # [batch, channels, height, width]
            output_binary = output[:, 0].cpu().numpy()
        elif len(output.shape) == 3:  # [batch, height, width]
            output_binary = output.cpu().numpy()
        else:
            print(f"Unexpected output shape: {output.shape}")
            continue

        print(f"Output binary shape: {output_binary.shape}")

        output_binary[output_binary > 0.5] = 1
        output_binary[output_binary <= 0.5] = 0
        
        # Remove batch dimension if present
        if len(output_binary.shape) == 3 and output_binary.shape[0] == 1:
            output_binary = output_binary[0]  # Remove batch dimension
        
        print(f"Final output shape: {output_binary.shape}")
        
        outputs.append(output_binary)
        end_ = time.time()
        print("post_process time {}".format(end_ - start_))
    end = time.time()
    print("Inference time {}".format(end - start))
    return outputs