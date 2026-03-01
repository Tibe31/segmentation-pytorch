"""Binary semantic segmentation module built on segmentation_models_pytorch.

This module provides :class:`SegmentationModel`, a PyTorch Lightning wrapper
that pairs any encoder-decoder architecture from the
`segmentation_models_pytorch <https://github.com/qubvel/segmentation_models.pytorch>`_
library with Dice loss and IoU metric tracking.  It is designed for
**binary** segmentation tasks (single foreground class).

Typical usage::

    from src.segmentation.models.unet import SegmentationModel

    model = SegmentationModel(
        arch="unet",
        encoder_name="resnet34",
        in_channels=3,
        out_classes=1,
        learning_rate=1e-3,
    )
"""

from __future__ import annotations

from typing import Any

import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
from torch import Tensor

__all__ = ["SegmentationModel"]

_ENCODER_DOWNSCALE_FACTOR = 32
"""int: Typical encoders have 5 down-sampling stages (2^5 = 32), so spatial
dimensions must be divisible by this factor to allow skip-connection
concatenation between encoder and decoder."""

_DEFAULT_THRESHOLD = 0.5
"""float: Probability threshold applied after sigmoid to binarise predicted
masks for metric computation."""


class SegmentationModel(pl.LightningModule):
    """PyTorch Lightning module for binary semantic segmentation.

    Wraps any architecture provided by *segmentation_models_pytorch*,
    computes Dice loss, and tracks per-image and dataset-level IoU metrics
    across training and validation epochs.

    All constructor arguments are persisted via
    :meth:`~pytorch_lightning.LightningModule.save_hyperparameters` and are
    therefore available under ``self.hparams`` and automatically restored by
    :meth:`load_from_checkpoint`.

    Attributes:
        model (torch.nn.Module): The underlying encoder-decoder network
            created by ``smp.create_model``.
        loss_fn (smp.losses.DiceLoss): Dice loss function operating on raw
            logits (``from_logits=True``).
    """

    def __init__(
        self,
        arch: str,
        encoder_name: str,
        in_channels: int,
        out_classes: int,
        learning_rate: float = 1e-3,
        **kwargs: Any,
    ) -> None:
        """Initialise the segmentation model.

        Args:
            arch: Architecture name accepted by ``smp.create_model``
                (e.g. ``"unet"``, ``"fpn"``, ``"deeplabv3+"``).
            encoder_name: Backbone encoder name accepted by
                *segmentation_models_pytorch*
                (e.g. ``"resnet34"``, ``"efficientnet-b0"``).
            in_channels: Number of input channels (e.g. ``3`` for RGB).
            out_classes: Number of output segmentation classes.  Use ``1``
                for binary segmentation.
            learning_rate: Learning rate for the Adam optimiser.
            **kwargs: Additional keyword arguments forwarded to
                ``smp.create_model`` (e.g. ``encoder_weights``).
        """
        super().__init__()
        self.save_hyperparameters()

        self.model = smp.create_model(
            arch,
            encoder_name=encoder_name,
            in_channels=in_channels,
            classes=out_classes,
            **kwargs,
        )
        self.loss_fn = smp.losses.DiceLoss(
            smp.losses.BINARY_MODE, from_logits=True
        )

        self._training_step_outputs: list[dict[str, Tensor]] = []
        self._validation_step_outputs: list[dict[str, Tensor]] = []

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: Tensor) -> Tensor:
        """Run the forward pass through the encoder-decoder network.

        Args:
            x: Input image batch of shape ``(B, C, H, W)``.

        Returns:
            Raw logits of shape ``(B, out_classes, H, W)``.
        """
        return self.model(x)

    # ------------------------------------------------------------------
    # Loss & metrics
    # ------------------------------------------------------------------

    def _compute_loss(self, logits: Tensor, targets: Tensor) -> Tensor:
        """Compute Dice loss between predictions and ground-truth masks.

        Both tensors are squeezed along ``dim=1`` to collapse the
        single-class channel before being passed to the loss function.

        Args:
            logits: Raw model output of shape ``(B, 1, H, W)``.
            targets: Ground-truth binary mask of shape ``(B, 1, H, W)``.

        Returns:
            Scalar loss value.
        """
        return self.loss_fn(logits.squeeze(1), targets.squeeze(1))

    @staticmethod
    def _compute_binary_stats(
        logits: Tensor,
        targets: Tensor,
        threshold: float = _DEFAULT_THRESHOLD,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Compute confusion-matrix statistics for binary segmentation.

        Logits are converted to probabilities via sigmoid, then binarised
        using *threshold* before counting true/false positive/negative
        pixels per image.

        Args:
            logits: Raw model output of shape ``(B, 1, H, W)``.
            targets: Ground-truth binary mask of shape ``(B, 1, H, W)``.
            threshold: Probability cut-off for the positive class.

        Returns:
            A 4-tuple ``(tp, fp, fn, tn)`` where each element is a
            ``Tensor`` of shape ``(B, 1)`` containing per-image counts.
        """
        pred_mask = (logits.sigmoid() > threshold).long()
        return smp.metrics.get_stats(pred_mask, targets.long(), mode="binary")

    # ------------------------------------------------------------------
    # Shared step / epoch-end logic
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_batch(image: Tensor, mask: Tensor) -> None:
        """Validate tensor shapes before the forward pass.

        Args:
            image: Image batch tensor.
            mask: Mask batch tensor.

        Raises:
            ValueError: If *image* or *mask* are not 4-D, or if the
                spatial dimensions of *image* are not divisible by
                :data:`_ENCODER_DOWNSCALE_FACTOR`.
        """
        if image.ndim != 4:
            raise ValueError(f"Expected 4-D image tensor, got {image.ndim}-D")
        h, w = image.shape[2:]
        if h % _ENCODER_DOWNSCALE_FACTOR or w % _ENCODER_DOWNSCALE_FACTOR:
            raise ValueError(
                f"Image dimensions ({h}x{w}) must be divisible by "
                f"{_ENCODER_DOWNSCALE_FACTOR}"
            )
        if mask.ndim != 4:
            raise ValueError(f"Expected 4-D mask tensor, got {mask.ndim}-D")

    def _shared_step(
        self, batch: tuple[Tensor, Tensor], stage: str
    ) -> dict[str, Tensor]:
        """Execute a single training or validation step.

        Performs input validation, a forward pass, loss computation, and
        confusion-matrix statistics collection.  The step-level loss is
        logged to the progress bar.

        Args:
            batch: A ``(image, mask)`` tuple produced by the dataloader.
            stage: Logging prefix — typically ``"train"`` or ``"valid"``.

        Returns:
            A dictionary with keys ``"loss"``, ``"tp"``, ``"fp"``,
            ``"fn"``, ``"tn"``.
        """
        image, mask = batch
        self._validate_batch(image, mask)

        logits = self(image)
        loss = self._compute_loss(logits, mask)
        self.log(
            f"{stage}_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=False,
        )

        tp, fp, fn, tn = self._compute_binary_stats(logits, mask)
        return {"loss": loss, "tp": tp, "fp": fp, "fn": fn, "tn": tn}

    def _shared_epoch_end(
        self, outputs: list[dict[str, Tensor]], stage: str
    ) -> None:
        """Aggregate step-level outputs and log epoch-level metrics.

        Two IoU variants are computed:

        * **per-image IoU** (``micro-imagewise``): IoU is calculated for
          each image independently and then averaged.  Sensitive to
          "empty" images (no foreground pixels).
        * **dataset IoU** (``micro``): intersection and union are
          accumulated across the whole dataset before computing IoU.
          More stable when the dataset contains empty images.

        Args:
            outputs: List of dictionaries returned by :meth:`_shared_step`
                over the epoch.
            stage: Logging prefix — typically ``"train"`` or ``"valid"``.
        """
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])

        per_image_iou = smp.metrics.iou_score(
            tp, fp, fn, tn, reduction="micro-imagewise"
        )
        dataset_iou = smp.metrics.iou_score(
            tp, fp, fn, tn, reduction="micro"
        )

        self.log_dict(
            {
                f"{stage}_per_image_iou": per_image_iou,
                f"{stage}_dataset_iou": dataset_iou,
                f"{stage}_loss": torch.stack([x["loss"] for x in outputs]).mean(),
            },
            prog_bar=True,
            logger=False,
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def training_step(
        self, batch: tuple[Tensor, Tensor], batch_idx: int
    ) -> dict[str, Tensor]:
        """Process a single training batch.

        The step output is cached in :attr:`_training_step_outputs` so
        that metrics can be aggregated at the end of the epoch by
        :meth:`on_train_epoch_end`.

        Args:
            batch: A ``(image, mask)`` tuple from the training dataloader.
            batch_idx: Index of the current batch (unused, required by
                Lightning).

        Returns:
            The dictionary produced by :meth:`_shared_step`.
        """
        step_output = self._shared_step(batch, "train")
        self._training_step_outputs.append(step_output)
        return step_output

    def on_train_epoch_end(self) -> None:
        """Aggregate training metrics and clear the step-output buffer.

        Called automatically by Lightning at the end of each training
        epoch.  Delegates to :meth:`_shared_epoch_end` and then releases
        the accumulated step outputs to free memory.
        """
        self._shared_epoch_end(self._training_step_outputs, "train")
        self._training_step_outputs.clear()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validation_step(
        self, batch: tuple[Tensor, Tensor], batch_idx: int
    ) -> dict[str, Tensor]:
        """Process a single validation batch.

        Mirrors :meth:`training_step` but uses the ``"valid"`` logging
        prefix and caches outputs in :attr:`_validation_step_outputs`.

        Args:
            batch: A ``(image, mask)`` tuple from the validation
                dataloader.
            batch_idx: Index of the current batch (unused, required by
                Lightning).

        Returns:
            The dictionary produced by :meth:`_shared_step`.
        """
        step_output = self._shared_step(batch, "valid")
        self._validation_step_outputs.append(step_output)
        return step_output

    def on_validation_epoch_end(self) -> None:
        """Aggregate validation metrics and clear the step-output buffer.

        Called automatically by Lightning at the end of each validation
        epoch.  Delegates to :meth:`_shared_epoch_end` and then releases
        the accumulated step outputs to free memory.
        """
        self._shared_epoch_end(self._validation_step_outputs, "valid")
        self._validation_step_outputs.clear()

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------

    def configure_optimizers(self) -> dict[str, Any]:
        """Configure the Adam optimiser.

        The learning rate is read from ``self.hparams.learning_rate``,
        which is set at construction time and persisted in checkpoints.

        Returns:
            A dictionary with key ``"optimizer"`` mapping to a
            :class:`torch.optim.Adam` instance, conforming to the
            Lightning optimiser configuration protocol.
        """
        return {
            "optimizer": torch.optim.Adam(
                self.parameters(), lr=self.hparams.learning_rate
            ),
        }
