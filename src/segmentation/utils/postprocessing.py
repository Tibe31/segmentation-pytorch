from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor

ResizeStrategy = Literal["resize_then_threshold", "threshold_then_resize"]


def postprocess_binary_mask(
    predictions: Tensor,
    output_size: tuple[int, int],
    threshold: float = 0.5,
    resize_strategy: ResizeStrategy = "resize_then_threshold",
    from_logits: bool = True,
) -> Tensor:
    """Convert model outputs into binary masks at the requested size.

    Args:
        predictions: Tensor of shape ``(B, 1, H, W)`` containing logits or
            probabilities.
        output_size: Target mask size as ``(height, width)``.
        threshold: Probability threshold for the positive class.
        resize_strategy: ``"resize_then_threshold"`` keeps smoother contours by
            resizing probabilities first. ``"threshold_then_resize"`` keeps
            pixelated edges by resizing a binary mask with nearest-neighbour.
        from_logits: Whether to apply sigmoid before thresholding.

    Returns:
        Binary float tensor of shape ``(B, 1, output_height, output_width)``.
    """
    if predictions.ndim != 4:
        raise ValueError(
            f"Expected predictions with shape (B, 1, H, W), got {tuple(predictions.shape)}"
        )

    probs = predictions.sigmoid() if from_logits else predictions

    if resize_strategy == "resize_then_threshold":
        resized = F.interpolate(
            probs,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )
        return (resized > threshold).to(dtype=torch.float32)

    if resize_strategy == "threshold_then_resize":
        binary = (probs > threshold).to(dtype=torch.float32)
        return F.interpolate(binary, size=output_size, mode="nearest")

    raise ValueError(f"Unsupported resize strategy: {resize_strategy}")
